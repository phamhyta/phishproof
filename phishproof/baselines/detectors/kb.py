"""Brand knowledge base rebuilt from Wikidata and Tranco.

Implements KnowPhish Sec. 3 (augmented BKB), Sec. 4.3 (category-based and popularity-based brand search), Sec. 4.4
(knowledge acquisition, logo variants, Whois domain variants, domain propagation), and App. A.1 and A.3 (eta = 50k and
the web-hosting exclusions). Every stage is cached under <kb_dir>/stages, so an interrupted build resumes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import queue
import re
import tempfile
import threading
import time
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from .domains import registrable_domain
from .logos import LogoBackend, LogoIndex

logger = logging.getLogger(__name__)

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
TRANCO_LATEST = "https://tranco-list.eu/top-1m.csv.zip"
GOOGLE_IMAGE_API = "https://www.googleapis.com/customsearch/v1"
ETA = 50_000
USER_AGENT = "phish-reimpl/0.1 (academic re-implementation of KnowPhish; contact: see README)"
BATCH = 200
PAGE = 5_000


@dataclass
class Brand:
    brand_id: str
    name: str
    aliases: list[str]
    domains: list[str]
    logos: list[str]


def alias_key(text: str) -> str:
    # "exact matching" (Sec. 5.3) ignores case and whitespace differences, since the paper gives no normalisation and LLM outputs vary in case.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class KnowledgeBase:
    """Augmented BKB with the AliasMap of KnowPhish Algorithm 1."""

    def __init__(self, brands: Iterable[Brand], logo_index: LogoIndex | None = None, root: Path | None = None) -> None:
        self.brands = {brand.brand_id: brand for brand in brands}
        self.logo_index = logo_index
        self.root = root
        self.alias_map: dict[str, set[str]] = defaultdict(set)
        for brand in self.brands.values():
            for alias in [brand.name, *brand.aliases]:
                key = alias_key(alias)
                if key:
                    self.alias_map[key].add(brand.brand_id)

    @classmethod
    def load(cls, kb_dir: str | Path) -> KnowledgeBase:
        root = Path(kb_dir)
        with (root / "brands.jsonl").open(encoding="utf-8") as handle:
            brands = [Brand(**json.loads(line)) for line in handle if line.strip()]
        index_path = root / "logo_index.npz"
        return cls(brands, LogoIndex.load(index_path) if index_path.is_file() else None, root)

    def brands_for_alias(self, text: str) -> set[str]:
        return set(self.alias_map.get(alias_key(text), set()))

    def __getitem__(self, brand_id: str) -> Brand:
        return self.brands[brand_id]

    def logo_path(self, relative: str) -> Path:
        if self.root is None:
            raise ValueError("knowledge base has no root folder")
        return self.root / relative


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _stage(stage_dir: Path, name: str, compute: Callable[[], Any]) -> Any:
    path = stage_dir / f"{name}.json"
    if path.is_file():
        logger.info("stage %s: loaded from cache", name)
        return json.loads(path.read_text(encoding="utf-8"))
    logger.info("stage %s: computing", name)
    value = compute()
    _atomic_write(path, json.dumps(value, ensure_ascii=False))
    return value


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _values(qids: Iterable[str]) -> str:
    return " ".join(f"wd:{qid}" for qid in qids)


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _retry_delay(response: requests.Response, attempt: int) -> float:
    try:
        return float(response.headers.get("Retry-After", ""))
    except ValueError:
        return min(5.0 * 2**attempt, 300.0)


class Sparql:
    """Wikidata query client that caches every result on disk."""

    def __init__(self, cache_dir: Path, endpoint: str = WDQS_ENDPOINT, pause: float = 1.0) -> None:
        self.cache_dir = cache_dir
        self.endpoint = endpoint
        self.pause = pause
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def select(self, query: str) -> list[dict[str, str]]:
        path = self.cache_dir / f"{hashlib.sha256(query.encode('utf-8')).hexdigest()}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        rows = self._fetch(query)
        _atomic_write(path, json.dumps(rows, ensure_ascii=False))
        return rows

    def _fetch(self, query: str) -> list[dict[str, str]]:
        for attempt in range(8):
            response = self.session.post(
                self.endpoint,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=120,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(_retry_delay(response, attempt))
                continue
            response.raise_for_status()
            time.sleep(self.pause)
            rows = []
            for binding in response.json()["results"]["bindings"]:
                row: dict[str, str] = {}
                for name, cell in binding.items():
                    row[name] = cell["value"]
                    if "xml:lang" in cell:
                        row[f"{name}_lang"] = cell["xml:lang"]
                rows.append(row)
            return rows
        raise RuntimeError("Wikidata query failed after 8 attempts")


def _paged(sparql: Sparql, query: str, page_size: int = PAGE) -> list[dict[str, str]]:
    # LIMIT/OFFSET paging without ORDER BY, because ordering these large classes exceeds the endpoint's time limit.
    rows: list[dict[str, str]] = []
    offset = 0
    while True:
        batch = sparql.select(f"{query} LIMIT {page_size} OFFSET {offset}")
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        offset += page_size


def load_tranco(path: str | Path, eta: int = ETA) -> dict[str, int]:
    """Map registrable domain to its best rank among the top eta entries (Sec. 4.3.2)."""
    ranks: dict[str, int] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            rank = int(row[0])
            if rank > eta:
                continue
            domain = registrable_domain(row[1])
            if domain:
                ranks[domain] = min(rank, ranks.get(domain, rank))
    return ranks


def download_tranco(out_path: Path, list_id: str | None = None) -> None:
    """Fetch a Tranco list; a list id pins a dated list, otherwise the latest list is fetched."""
    url = f"https://tranco-list.eu/download/{list_id}/1000000" if list_id else TRANCO_LATEST
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=300)
    response.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if response.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            out_path.write_bytes(archive.read(archive.namelist()[0]))
    else:
        out_path.write_bytes(response.content)


def resolve_categories(sparql: Sparql, entries: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Resolve category labels to Wikidata classes, verifying any pinned QID against its English label."""
    resolved: list[dict[str, Any]] = []
    for entry in entries:
        label = entry["label"]
        escaped = label.replace('"', '\\"')
        if entry.get("qid"):
            qid = entry["qid"]
            rows = sparql.select(f'SELECT ?l WHERE {{ wd:{qid} rdfs:label ?l . FILTER(LANG(?l) = "en") }}')
            actual = rows[0]["l"] if rows else ""
            if actual.casefold() != label.casefold():
                raise ValueError(f"pinned {qid} has English label {actual!r}, expected {label!r}")
            resolved.append({"label": label, "qid": qid, "candidates": [qid]})
            continue
        rows = sparql.select(
            f'SELECT DISTINCT ?c ?n WHERE {{ ?c rdfs:label "{escaped}"@en ; wdt:P279 ?super ; '
            "wikibase:sitelinks ?n . } ORDER BY DESC(?n) LIMIT 5"
        )
        if not rows:
            logger.warning("category %r did not resolve to a Wikidata class; skipped", label)
            continue
        # an unpinned label resolves to the same-label class with the most sitelinks, because the paper does not list its category QIDs.
        resolved.append({"label": label, "qid": _qid(rows[0]["c"]), "candidates": [_qid(r["c"]) for r in rows]})
    return resolved


def category_brands(sparql: Sparql, categories: list[dict[str, Any]]) -> set[str]:
    """Return B_n: instances of each narrow category and of its direct subclasses (Sec. 4.3.1)."""
    brands: set[str] = set()
    for category in categories:
        qid = category["qid"]
        rows = _paged(
            sparql, f"SELECT DISTINCT ?b WHERE {{ {{ ?b wdt:P31 wd:{qid} }} UNION {{ ?c wdt:P279 wd:{qid} . ?b wdt:P31 ?c }} }}"
        )
        brands.update(_qid(row["b"]) for row in rows)
    return brands


def popularity_brands(sparql: Sparql, categories: list[dict[str, Any]], tranco: dict[str, int], eta: int = ETA) -> set[str]:
    """Return B_g: direct instances of each general category whose best domain rank is at most eta (Sec. 4.3.2)."""
    best_rank: dict[str, int] = {}
    for category in categories:
        qid = category["qid"]
        rows = _paged(sparql, f"SELECT DISTINCT ?b ?site WHERE {{ ?b wdt:P31 wd:{qid} ; wdt:P856 ?site }}")
        for row in rows:
            rank = tranco.get(registrable_domain(row["site"]))
            if rank is not None:
                brand = _qid(row["b"])
                best_rank[brand] = min(rank, best_rank.get(brand, rank))
    return {brand for brand, rank in best_rank.items() if rank <= eta}


def acquire(sparql: Sparql, brand_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch names, aliases, official websites and logo files (Sec. 4.4.1)."""
    records: dict[str, dict[str, Any]] = {
        qid: {"name": "", "aliases": set(), "domains": set(), "logo_urls": set()} for qid in brand_ids
    }
    english: dict[str, str] = {}
    for batch in _chunks(brand_ids, BATCH):
        rows = sparql.select(
            f"SELECT ?b ?logo ?site WHERE {{ VALUES ?b {{ {_values(batch)} }} "
            "OPTIONAL { ?b wdt:P154 ?logo } OPTIONAL { ?b wdt:P856 ?site } }"
        )
        for row in rows:
            record = records[_qid(row["b"])]
            if "logo" in row:
                record["logo_urls"].add(row["logo"])
            if "site" in row:
                domain = registrable_domain(row["site"])
                if domain:
                    record["domains"].add(domain)
        # aliases are rdfs:label plus skos:altLabel in every language, since Sec. 4.4.1 says "alternative names in different languages" and Fig. 5 shows both kinds.
        rows = sparql.select(
            f"SELECT ?b ?label WHERE {{ VALUES ?b {{ {_values(batch)} }} "
            "{ ?b rdfs:label ?label } UNION { ?b skos:altLabel ?label } }"
        )
        for row in rows:
            qid = _qid(row["b"])
            records[qid]["aliases"].add(row["label"])
            if row.get("label_lang") == "en" and (qid not in english or len(row["label"]) < len(english[qid])):
                english[qid] = row["label"]
    output: dict[str, dict[str, Any]] = {}
    for qid, record in records.items():
        aliases = sorted(record["aliases"])
        # the brand name is the shortest English label, falling back to the first alias.
        output[qid] = {
            "name": english.get(qid) or (aliases[0] if aliases else ""),
            "aliases": aliases,
            "domains": sorted(record["domains"]),
            "logo_urls": sorted(record["logo_urls"]),
        }
    return output


def web_hosting_brands(sparql: Sparql, brand_ids: list[str], categories: list[dict[str, Any]]) -> set[str]:
    """Return brands to exclude because they are direct instances of a web-hosting category (App. A.3)."""
    if not categories:
        return set()
    excluded: set[str] = set()
    classes = _values(category["qid"] for category in categories)
    for batch in _chunks(brand_ids, BATCH):
        rows = sparql.select(f"SELECT ?b WHERE {{ VALUES ?b {{ {_values(batch)} }} VALUES ?c {{ {classes} }} ?b wdt:P31 ?c }}")
        excluded.update(_qid(row["b"]) for row in rows)
    return excluded


def subsidiary_pairs(sparql: Sparql, brand_ids: list[str]) -> list[list[str]]:
    """Return owned_by (P127) and parent_organization (P749) edges between brands in the list (Sec. 4.4.2)."""
    known = set(brand_ids)
    pairs: list[list[str]] = []
    for batch in _chunks(brand_ids, BATCH):
        rows = sparql.select(
            f"SELECT ?a ?p WHERE {{ VALUES ?a {{ {_values(batch)} }} {{ ?a wdt:P127 ?p }} UNION {{ ?a wdt:P749 ?p }} }}"
        )
        for row in rows:
            parent = _qid(row["p"])
            if parent in known:
                pairs.append([_qid(row["a"]), parent])
    return pairs


def propagate_domains(domains: dict[str, list[str]], pairs: list[list[str]]) -> dict[str, list[str]]:
    """Add the domains of 1-hop subsidiary neighbours, read from a snapshot so that nothing cascades (Sec. 4.4.2)."""
    neighbours: dict[str, set[str]] = defaultdict(set)
    for left, right in pairs:
        if left in domains and right in domains and left != right:
            neighbours[left].add(right)
            neighbours[right].add(left)
    return {
        brand: sorted(set(own).union(*(domains[other] for other in neighbours[brand])))
        for brand, own in domains.items()
    }


_PLACEHOLDER_ORG = re.compile(
    r"redacted|privacy|proxy|whoisguard|not disclosed|withheld|gdpr|masked|data protected|registration private|statutory",
)


def canonical_org(value: Any) -> str | None:
    """Return a comparable Whois organisation, or None for empty and privacy-service values."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    text = " ".join(value.casefold().split())
    # privacy-service placeholders never count as a shared owner, although Sec. 4.4.2 only says organisation details must be identical.
    if not text or _PLACEHOLDER_ORG.search(text):
        return None
    return text


def whois_organisations(domains: list[str], path: Path, workers: int, pause: float) -> dict[str, str | None]:
    """Look up each domain's Whois organisation, appending results to a single resumable JSONL file."""
    import whois

    known: dict[str, str | None] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    known[row["domain"]] = row["org"]
    lock = threading.Lock()
    path.parent.mkdir(parents=True, exist_ok=True)

    def lookup(domain: str) -> None:
        try:
            org = canonical_org(whois.whois(domain).get("org"))
        except Exception as exc:
            logger.debug("whois failed for %s: %s", domain, exc)
            org = None
        with lock:
            known[domain] = org
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"domain": domain, "org": org}, ensure_ascii=False) + "\n")
        time.sleep(pause)

    todo = [domain for domain in domains if domain not in known]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lookup, todo))
    return known


def expand_domains_by_whois(
    domains: dict[str, list[str]], orgs: dict[str, str | None], tranco: Iterable[str]
) -> dict[str, list[str]]:
    """Add Tranco domains that share a Whois organisation with a brand's domains (Sec. 4.4.2)."""
    by_org: dict[str, set[str]] = defaultdict(set)
    for domain in tranco:
        org = orgs.get(domain)
        if org:
            by_org[org].add(domain)
    expanded: dict[str, list[str]] = {}
    for brand, own in domains.items():
        extra: set[str] = set()
        for domain in own:
            org = orgs.get(domain)
            if org:
                extra |= by_org[org]
        expanded[brand] = sorted(set(own) | extra)
    return expanded


def _save_logo(content: bytes, path: Path) -> bool:
    """Save an image as a PNG on white, because Phishpedia's RGB conversion turns transparency black."""
    try:
        with Image.open(io.BytesIO(content)) as image:
            rgba = image.convert("RGBA")
    except Exception:
        return False
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(path, format="PNG")
    return True


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def wikidata_logos(records: dict[str, dict[str, Any]], kb_dir: Path, pause: float) -> dict[str, list[str]]:
    """Download Wikidata logo images (P154) as raster thumbnails, because many are SVG."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    output: dict[str, list[str]] = {}
    for qid, record in records.items():
        files: list[str] = []
        for url in record["logo_urls"]:
            relative = f"logos/{qid}/wikidata_{_digest(url)}.png"
            target = kb_dir / relative
            if not target.is_file():
                # Commons renders logos at 300 px wide, because the paper gives no size.
                try:
                    response = session.get(url, params={"width": 300}, timeout=60)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    logger.debug("logo download failed for %s: %s", url, exc)
                    continue
                time.sleep(pause)
                if not _save_logo(response.content, target):
                    continue
            files.append(relative)
        output[qid] = files
    return output


def _capture_worker(jobs: queue.Queue[str], shots_dir: Path, timeout_ms: int) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800}, ignore_https_errors=True)
        page = context.new_page()
        while True:
            try:
                domain = jobs.get_nowait()
            except queue.Empty:
                break
            shot = shots_dir / f"{domain}.png"
            try:
                page.goto(f"https://{domain}", timeout=timeout_ms, wait_until="load")
                page.screenshot(path=str(shot))
            except Exception as exc:
                logger.debug("capture failed for %s: %s", domain, exc)
        browser.close()


def homepage_logos(
    records: dict[str, dict[str, Any]], kb_dir: Path, backend: LogoBackend, workers: int, timeout_ms: int = 30_000
) -> dict[str, list[str]]:
    """Capture each brand's official domains and detect their logos: DetectLogos in Sec. 4.4.2 (brand sites only)."""
    shots_dir = kb_dir / "homepages"
    shots_dir.mkdir(parents=True, exist_ok=True)
    domains = sorted({domain for record in records.values() for domain in record["domains"]})
    jobs: queue.Queue[str] = queue.Queue()
    for domain in domains:
        if not (shots_dir / f"{domain}.png").is_file():
            jobs.put(domain)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_capture_worker, jobs, shots_dir, timeout_ms) for _ in range(workers)]
        for future in futures:
            future.result()
    output: dict[str, list[str]] = {}
    for qid, record in records.items():
        files: list[str] = []
        for domain in record["domains"]:
            shot = shots_dir / f"{domain}.png"
            if not shot.is_file():
                continue
            relative = f"logos/{qid}/homepage_{domain}.png"
            target = kb_dir / relative
            if not target.is_file():
                logo = backend.top_logo(shot)
                if logo is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                logo.convert("RGB").save(target, format="PNG")
            files.append(relative)
        output[qid] = files
    return output


def google_image_logos(
    records: dict[str, dict[str, Any]], kb_dir: Path, api_key: str, engine_id: str, pause: float
) -> dict[str, list[str]]:
    """Search '<name> logo' and keep images hosted on the brand's domains (Sec. 4.4.2)."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    output: dict[str, list[str]] = {}
    for qid, record in records.items():
        results_path = kb_dir / "stages" / "google" / f"{qid}.json"
        if results_path.is_file():
            links = json.loads(results_path.read_text(encoding="utf-8"))
        else:
            params = {
                "key": api_key,
                "cx": engine_id,
                "q": f"{record['name']} logo",
                "searchType": "image",
                "num": 10,
            }
            try:
                response = session.get(GOOGLE_IMAGE_API, params=params, timeout=60)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("image search failed for %s: %s", qid, exc)
                output[qid] = []
                continue
            links = [item["link"] for item in response.json().get("items", [])]
            _atomic_write(results_path, json.dumps(links))
            time.sleep(pause)
        allowed = set(record["domains"])
        files: list[str] = []
        for link in links:
            if registrable_domain(link) not in allowed:
                continue
            relative = f"logos/{qid}/google_{_digest(link)}.png"
            target = kb_dir / relative
            if not target.is_file():
                try:
                    image = session.get(link, timeout=60)
                    image.raise_for_status()
                except requests.RequestException:
                    continue
                if not _save_logo(image.content, target):
                    continue
            files.append(relative)
        output[qid] = files
    return output


@dataclass
class BuildConfig:
    kb_dir: Path
    categories: Path
    tranco_csv: Path | None = None
    tranco_list_id: str | None = None
    eta: int = ETA
    skip_whois: bool = False
    skip_homepage_logos: bool = False
    skip_google_images: bool = False
    whois_workers: int = 8
    capture_workers: int = 8
    pause: float = 0.2


def _check_prerequisites(config: BuildConfig) -> tuple[str, str]:
    """Stop before any long stage when a stage that was not skipped cannot run."""
    if not config.skip_whois:
        import whois  # noqa: F401
    if not config.skip_homepage_logos:
        import playwright  # noqa: F401
    if config.skip_google_images:
        return "", ""
    api_key, engine_id = os.environ.get("GOOGLE_API_KEY", ""), os.environ.get("GOOGLE_CSE_ID", "")
    if not api_key or not engine_id:
        raise SystemExit("Set GOOGLE_API_KEY and GOOGLE_CSE_ID, or pass --skip-google-images (a recorded deviation).")
    return api_key, engine_id


def build_kb(config: BuildConfig, backend: LogoBackend) -> dict[str, Any]:
    """Run KnowPhish Sec. 4.3-4.4 and write brands.jsonl, logo_index.npz and meta.json."""
    google_key, google_engine = _check_prerequisites(config)
    kb_dir = config.kb_dir
    stages = kb_dir / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    tranco_csv = config.tranco_csv or (stages / "tranco.csv")
    if not tranco_csv.is_file():
        download_tranco(tranco_csv, config.tranco_list_id)
    tranco = load_tranco(tranco_csv, config.eta)
    sparql = Sparql(stages / "sparql")
    wanted = json.loads(config.categories.read_text(encoding="utf-8"))

    categories = _stage(
        stages, "categories", lambda: {name: resolve_categories(sparql, wanted[name]) for name in ("narrow", "general", "exclude")}
    )
    narrow = _stage(stages, "brands_narrow", lambda: sorted(category_brands(sparql, categories["narrow"])))
    general = _stage(
        stages, "brands_general", lambda: sorted(popularity_brands(sparql, categories["general"], tranco, config.eta))
    )
    brand_ids = sorted(set(narrow) | set(general))
    acquired = _stage(stages, "acquired", lambda: acquire(sparql, brand_ids))
    excluded = set(_stage(stages, "excluded", lambda: sorted(web_hosting_brands(sparql, brand_ids, categories["exclude"]))))
    # brands with no official website are dropped, because the domain check needs at least one legitimate domain.
    records = {q: r for q, r in acquired.items() if q not in excluded and r["name"] and r["domains"]}

    sources = [_stage(stages, "logos_wikidata", lambda: wikidata_logos(records, kb_dir, config.pause))]
    if not config.skip_homepage_logos:
        sources.append(_stage(stages, "logos_homepage", lambda: homepage_logos(records, kb_dir, backend, config.capture_workers)))
    if not config.skip_google_images:
        sources.append(
            _stage(stages, "logos_google", lambda: google_image_logos(records, kb_dir, google_key, google_engine, config.pause))
        )

    domains: dict[str, list[str]] = {qid: list(record["domains"]) for qid, record in records.items()}
    if not config.skip_whois:
        everything = sorted(set(tranco) | {domain for values in domains.values() for domain in values})
        orgs = whois_organisations(everything, stages / "whois.jsonl", config.whois_workers, config.pause)
        domains = expand_domains_by_whois(domains, orgs, tranco)
    pairs = _stage(stages, "subsidiaries", lambda: subsidiary_pairs(sparql, sorted(records)))
    domains = propagate_domains(domains, pairs)

    brands = [
        Brand(
            brand_id=qid,
            name=record["name"],
            aliases=record["aliases"],
            domains=domains[qid],
            logos=sorted({path for source in sources for path in source.get(qid, [])}),
        )
        for qid, record in sorted(records.items())
    ]
    with (kb_dir / "brands.jsonl").open("w", encoding="utf-8") as handle:
        for brand in brands:
            handle.write(json.dumps(asdict(brand), ensure_ascii=False) + "\n")

    vectors: list[np.ndarray] = []
    owners: list[str] = []
    for brand in brands:
        for relative in brand.logos:
            with Image.open(kb_dir / relative) as image:
                vectors.append(backend.embed(image.convert("RGB")))
            owners.append(brand.brand_id)
    matrix = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, 1), dtype=np.float32)
    LogoIndex(matrix, owners).save(kb_dir / "logo_index.npz")

    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_brands": len(brands),
        "n_brands_with_logo": sum(1 for brand in brands if brand.logos),
        "n_logos": len(owners),
        "n_narrow": len(narrow),
        "n_general": len(general),
        "n_excluded": len(excluded),
        "eta": config.eta,
        "tranco_sha256": hashlib.sha256(tranco_csv.read_bytes()).hexdigest(),
        "tranco_list_id": config.tranco_list_id,
        "categories_sha256": hashlib.sha256(config.categories.read_bytes()).hexdigest(),
        "skipped": {
            "whois": config.skip_whois,
            "homepage_logos": config.skip_homepage_logos,
            "google_images": config.skip_google_images,
        },
    }
    _atomic_write(kb_dir / "meta.json", json.dumps(meta, indent=2))
    if not 15_000 <= len(brands) <= 26_000:
        logger.warning("built %d brands; KnowPhish reports 20,514, so review data/categories.json", len(brands))
    return meta
