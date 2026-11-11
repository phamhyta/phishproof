"""CLIP encoders, contrastive fine-tuning and inference for PhishAgent's multimodal retriever.

Implements PhishAgent Sec. 4.3 (webpage and brand encoding, Eq. 1, dot-product inference, the contrastive loss) and
App. A.1 and A.3 (unit weights; batch size 4, learning rate 2e-6, 111 negatives, Adam, 10 epochs, best validation
checkpoint). This module needs the "retriever" extra.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .base import Page
from .kb import KnowledgeBase
from .logos import LogoBackend
from .pagetext import plain_text
from .retriever import (
    BrandEntry,
    Hit,
    Pair,
    brand_entries,
    build_pairs,
    ground_pages,
    ranked_brands,
    split_pairs,
)

logger = logging.getLogger(__name__)

# ViT-B-32-quickgelu with OpenAI weights, the CLIP used elsewhere in the paper, because the paper does not name a CLIP variant.
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"


@dataclass
class TrainConfig:
    n_negatives: int = 111
    rounds: int = 2
    batch_size: int = 4
    learning_rate: float = 2e-6
    epochs: int = 10
    val_fraction: float = 0.1
    seed: int = 13
    data_workers: int = 4
    # the softmax scores are multiplied by the model's fixed logit scale when this is None, because the paper's loss shows no scale and unscaled cosines give almost no gradient.
    logit_scale: float | None = None


def combine(
    model: Any, images: torch.Tensor, tokens: torch.Tensor, has_logo: torch.Tensor
) -> torch.Tensor:
    """Return Eq. 1 with unit weights; the image term is dropped where there is no logo."""
    text = F.normalize(model.encode_text(tokens), dim=-1)
    image = F.normalize(model.encode_image(images), dim=-1)
    # each modality is L2-normalised before the weighted sum, because the paper only says encodings are "added together, and normalized".
    return F.normalize(text + image * has_logo.unsqueeze(-1), dim=-1)


class _PairDataset(Dataset):
    """Load the page logo and the 1 + N brand logos of each training pair."""

    def __init__(
        self,
        pairs: list[Pair],
        pages: dict[str, tuple[str, str | None]],
        entries: list[BrandEntry],
        kb_root: Path,
        preprocess: Any,
    ) -> None:
        self.pairs = pairs
        self.pages = pages
        self.entries = entries
        self.kb_root = kb_root
        self.preprocess = preprocess
        self.blank = preprocess(Image.new("RGB", (224, 224), (255, 255, 255)))

    def __len__(self) -> int:
        return len(self.pairs)

    def _image(self, path: str | Path | None) -> tuple[torch.Tensor, float]:
        if path is None:
            return self.blank, 0.0
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB")), 1.0

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        text, crop = self.pages[pair.page_id]
        page_image, page_has_logo = self._image(crop)
        candidates = [pair.positive, *pair.negatives]
        images, flags, texts = [], [], []
        for candidate in candidates:
            entry = self.entries[candidate]
            image, flag = self._image(self.kb_root / entry.logo if entry.logo else None)
            images.append(image)
            flags.append(flag)
            texts.append(entry.text)
        return {
            "page_text": text,
            "page_image": page_image,
            "page_has_logo": page_has_logo,
            "brand_texts": texts,
            "brand_images": torch.stack(images),
            "brand_has_logo": torch.tensor(flags, dtype=torch.float32),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "page_texts": [item["page_text"] for item in batch],
        "page_images": torch.stack([item["page_image"] for item in batch]),
        "page_has_logo": torch.tensor([item["page_has_logo"] for item in batch], dtype=torch.float32),
        "brand_texts": [text for item in batch for text in item["brand_texts"]],
        "brand_images": torch.cat([item["brand_images"] for item in batch]),
        "brand_has_logo": torch.cat([item["brand_has_logo"] for item in batch]),
    }


def _scores(model: Any, tokenizer: Any, batch: dict[str, Any], n_candidates: int, device: str) -> torch.Tensor:
    """Return s_0..s_N of Sec. 4.3 Training for each page in the batch."""
    pages = combine(
        model,
        batch["page_images"].to(device),
        tokenizer(batch["page_texts"]).to(device),
        batch["page_has_logo"].to(device),
    )
    brands = combine(
        model,
        batch["brand_images"].to(device),
        tokenizer(batch["brand_texts"]).to(device),
        batch["brand_has_logo"].to(device),
    ).view(pages.shape[0], n_candidates, -1)
    return torch.einsum("bd,bnd->bn", pages, brands)


def _accuracy(model: Any, tokenizer: Any, loader: DataLoader, n_candidates: int, device: str) -> float:
    # the "validation score" for checkpoint selection is top-1 accuracy among the 1+N candidates, because the paper does not define it.
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            predictions = _scores(model, tokenizer, batch, n_candidates, device).argmax(dim=1)
            correct += int((predictions == 0).sum().item())
            total += predictions.numel()
    return correct / max(total, 1)


def _page_inputs(
    pages: list[Page], logos: LogoBackend, crops_dir: Path
) -> dict[str, tuple[str, str | None]]:
    """Return each page's processed text and the path of its detected logo crop, if any."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, tuple[str, str | None]] = {}
    for page in pages:
        html = page.read_html() or ""
        crop_path: str | None = None
        if page.has_screenshot():
            logo = logos.top_logo(page.screenshot_path)
            if logo is not None:
                target = crops_dir / f"{hashlib.sha1(page.page_id.encode('utf-8')).hexdigest()}.png"
                logo.convert("RGB").save(target, format="PNG")
                crop_path = str(target)
        inputs[page.page_id] = (plain_text(html), crop_path)
    return inputs


def _load_clip(device: str) -> tuple[Any, Any, Any]:
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    return model.to(device), preprocess, open_clip.get_tokenizer(MODEL_NAME)


def build_index(model: Any, tokenizer: Any, preprocess: Any, kb: KnowledgeBase, entries: list[BrandEntry], device: str, batch_size: int = 256) -> np.ndarray:
    """Encode every brand entry with the selected checkpoint (Sec. 4.3 Brand Encoding)."""
    model.eval()
    if kb.root is None:
        raise ValueError("knowledge base has no root folder")
    blank = preprocess(Image.new("RGB", (224, 224), (255, 255, 255)))
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(entries), batch_size):
            chunk = entries[start : start + batch_size]
            images, flags = [], []
            for entry in chunk:
                if entry.logo is None:
                    images.append(blank)
                    flags.append(0.0)
                    continue
                with Image.open(kb.root / entry.logo) as image:
                    images.append(preprocess(image.convert("RGB")))
                flags.append(1.0)
            encoded = combine(
                model,
                torch.stack(images).to(device),
                tokenizer([entry.text for entry in chunk]).to(device),
                torch.tensor(flags, dtype=torch.float32, device=device),
            )
            vectors.append(encoded.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors)


def finetune(
    config: TrainConfig,
    training_pages: list[Page],
    labels: dict[str, str],
    kb: KnowledgeBase,
    logos: LogoBackend,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Train the retriever as PhishAgent App. A.3 describes and save best.pt, index.npz and train_meta.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if kb.root is None:
        raise ValueError("knowledge base has no root folder")
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grounded = ground_pages(training_pages, labels, kb)
    if not grounded:
        raise ValueError("no training page could be grounded in the knowledge base")
    entries = brand_entries(kb)
    pairs = build_pairs(grounded, entries, config.n_negatives, config.rounds, config.seed)
    train_pairs, val_pairs = split_pairs(pairs, config.val_fraction, config.seed)
    logger.info("%d grounded pages, %d train pairs, %d validation pairs", len(grounded), len(train_pairs), len(val_pairs))

    page_inputs = _page_inputs([page for page, _ in grounded], logos, out_dir / "crops")
    model, preprocess, tokenizer = _load_clip(device)
    n_candidates = config.n_negatives + 1
    loader_args: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.data_workers,
        "collate_fn": _collate,
    }
    train_loader = DataLoader(
        _PairDataset(train_pairs, page_inputs, entries, kb.root, preprocess), shuffle=True, **loader_args
    )
    val_loader = DataLoader(_PairDataset(val_pairs, page_inputs, entries, kb.root, preprocess), **loader_args)

    # the logit scale stays frozen during fine-tuning, because the paper does not mention it.
    model.logit_scale.requires_grad_(False)
    scale = float(model.logit_scale.exp().item()) if config.logit_scale is None else config.logit_scale
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    best_score, best_epoch = -1.0, -1
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        running, steps = 0.0, 0
        for batch in train_loader:
            scores = _scores(model, tokenizer, batch, n_candidates, device) * scale
            loss = F.cross_entropy(scores, torch.zeros(scores.shape[0], dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        score = _accuracy(model, tokenizer, val_loader, n_candidates, device)
        history.append({"epoch": epoch, "train_loss": running / max(steps, 1), "val_accuracy": score})
        logger.info("epoch %d: loss %.4f, validation accuracy %.4f", epoch, running / max(steps, 1), score)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
    matrix = build_index(model, tokenizer, preprocess, kb, entries, device)
    np.savez(out_dir / "index.npz", embeddings=matrix, brand_ids=np.array([e.brand_id for e in entries], dtype=str))
    meta = {
        **asdict(config),
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "logit_scale_used": scale,
        "n_training_pages": len(training_pages),
        "n_grounded_pages": len(grounded),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_score,
        "history": history,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class ClipRetriever:
    """Top-k retrieval over the fine-tuned index (Sec. 4.3 Inference)."""

    def __init__(self, retriever_dir: str | Path) -> None:
        directory = Path(retriever_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess, self.tokenizer = _load_clip(self.device)
        state = torch.load(directory / "best.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        with np.load(directory / "index.npz", allow_pickle=False) as data:
            self.embeddings = data["embeddings"]
            self.brand_ids = [str(value) for value in data["brand_ids"].tolist()]
        self._blank = self.preprocess(Image.new("RGB", (224, 224), (255, 255, 255)))
        self._lock = threading.Lock()

    def search(self, text: str, logo: Image.Image | None, limit: int) -> list[Hit]:
        """Return the best distinct brands; the CLIP tokenizer truncates long text to its 77-token context."""
        image = self._blank if logo is None else self.preprocess(logo.convert("RGB"))
        flag = 0.0 if logo is None else 1.0
        with self._lock, torch.no_grad():
            query = combine(
                self.model,
                image.unsqueeze(0).to(self.device),
                self.tokenizer([text]).to(self.device),
                torch.tensor([flag], dtype=torch.float32, device=self.device),
            )
        scores = self.embeddings @ query.cpu().numpy()[0]
        return ranked_brands(scores, self.brand_ids, limit)
