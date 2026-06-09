"""RQ2 evidence-targeted perturbations + flip rate (experiments.tex §exp_rq2).

Each perturbation targets a cited evidence cue while PRESERVING the page's true label,
so a label flip measures reason fragility, not a changed ground truth:
  - form-action cloak: rewrite form actions to an on-brand-looking domain (attacks the
    form_action_domain cue)
  - logo morph: blur/scramble the rendered logo region (attacks the logo_brand cue)

flip_rate = fraction of pages whose panel verdict changes under the perturbation.
Perturbed pages are written under out_dir; the panel is re-run on them by the runner.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageFilter

from ..schema import Label, PageRecord
from ..tools.brands import canonical_brand
from ..tools.logo_brand import parse_logo_boxes


def _onbrand_domain(page: PageRecord) -> str:
    brand = canonical_brand(page.brand) or "secure-login"
    return f"{brand.replace(' ', '')}.com"


def cloak_form_action(page: PageRecord, out_dir: Path) -> PageRecord:
    """Rewrite every form action to an on-brand-looking domain (cloak the cue)."""
    if not page.dom_html_path or not Path(page.dom_html_path).exists():
        return page
    soup = BeautifulSoup(Path(page.dom_html_path).read_text(errors="replace"), "lxml")
    target = f"https://{_onbrand_domain(page)}/login"
    for form in soup.find_all("form"):
        form["action"] = target
    out_dir.mkdir(parents=True, exist_ok=True)
    new_html = out_dir / f"{page.page_id}_cloak.html"
    new_html.write_text(str(soup), encoding="utf-8")
    return page.model_copy(update={"dom_html_path": str(new_html),
                                   "page_id": page.page_id + "::cloak"})


def morph_logo(page: PageRecord, out_dir: Path) -> PageRecord:
    """Blur the highest-confidence logo region in the screenshot (morph the cue)."""
    if not page.screenshot_path or not Path(page.screenshot_path).exists():
        return page
    boxes = parse_logo_boxes(page)
    if not boxes:
        return page
    img = Image.open(page.screenshot_path).convert("RGB")
    (x1, y1, x2, y2), _ = boxes[0]
    region = img.crop((int(x1), int(y1), int(x2), int(y2)))
    region = region.filter(ImageFilter.GaussianBlur(8))
    img.paste(region, (int(x1), int(y1)))
    out_dir.mkdir(parents=True, exist_ok=True)
    new_png = out_dir / f"{page.page_id}_morph.png"
    img.save(new_png)
    return page.model_copy(update={"screenshot_path": str(new_png),
                                   "page_id": page.page_id + "::morph"})


def occlude_logo(page: PageRecord, out_dir: Path) -> PageRecord:
    """Destroy the brand logo: overwrite its region with mid-gray + noise.

    Stronger than ``morph_logo``'s blur -- it removes the visual brand cue entirely,
    synthesizing the *logo-free / minimal-cue* phishing the Phishpedia corpus lacks. The
    page stays phishing (off-brand host, credential form), but the perceptual brand
    evidence the logo-grounding tool checks is gone, so a cited logo cue should fail its
    CLIP re-derivation and groundedness should fall.
    """
    if not page.screenshot_path or not Path(page.screenshot_path).exists():
        return page
    boxes = parse_logo_boxes(page)
    if not boxes:
        return page
    img = Image.open(page.screenshot_path).convert("RGB")
    (x1, y1, x2, y2), _ = boxes[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    noise = (np.random.default_rng(0).integers(110, 150, size=(h, w, 3))).astype("uint8")
    img.paste(Image.fromarray(noise), (x1, y1))
    out_dir.mkdir(parents=True, exist_ok=True)
    new_png = out_dir / f"{page.page_id}_occlude.png"
    img.save(new_png)
    return page.model_copy(update={"screenshot_path": str(new_png),
                                   "page_id": page.page_id + "::occlude"})


def strip_brand_text(page: PageRecord, out_dir: Path) -> PageRecord:
    """White-box: erase the brand's textual identity from the DOM (title, meta, headings,
    visible text), so the page-content brand grounder can no longer re-derive the brand.

    Combined with a form-action cloak and a logo occlusion, this attacks every active
    verifier at once -- the adaptive adversary that knows the grounding tools and tries to
    leave no cue that grounds, while the page still collects credentials.
    """
    if not page.dom_html_path or not Path(page.dom_html_path).exists():
        return page
    brand = canonical_brand(page.brand)
    soup = BeautifulSoup(Path(page.dom_html_path).read_text(errors="replace"), "lxml")
    tokens = [t for t in (brand or "").split() if len(t) >= 3] or ([brand] if brand else [])

    def scrub(text: str) -> str:
        out = text
        for t in tokens:
            out = re.sub(re.escape(t), "", out, flags=re.IGNORECASE)
        return out

    if soup.title and soup.title.string:
        soup.title.string = scrub(soup.title.string)
    for m in soup.find_all("meta"):
        if m.get("content"):
            m["content"] = scrub(m["content"])
    for el in soup.find_all(["h1", "h2", "h3"]):
        el.string = scrub(el.get_text(" "))
    for node in soup.find_all(string=True):
        if tokens and any(t.lower() in node.lower() for t in tokens):
            node.replace_with(scrub(str(node)))
    out_dir.mkdir(parents=True, exist_ok=True)
    new_html = out_dir / f"{page.page_id}_stripbrand.html"
    new_html.write_text(str(soup), encoding="utf-8")
    return page.model_copy(update={"dom_html_path": str(new_html),
                                   "page_id": page.page_id + "::stripbrand"})


def flip_rate(orig: list[Label], perturbed: list[Label]) -> float:
    """Fraction of pages whose verdict changed under perturbation."""
    if not orig:
        return 0.0
    flips = sum(a != b for a, b in zip(orig, perturbed))
    return flips / len(orig)
