"""Perceptual logo-brand grounding (returns similarity in [0,1]).

Crops the rendered logo from the screenshot (using the YOLO logo boxes Phishpedia
ships in yolo_coords.txt) and matches it to the claimed brand with CLIP zero-shot
text-image similarity — so no per-brand reference-logo targetlist is needed.

CLIP (open_clip + torch) is imported lazily: cropping is testable with PIL alone, and
the embedder is only loaded when a logo cue is actually grounded (Phase 3 run on Mac CPU).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..schema import Cue, PageRecord

Box = tuple[float, float, float, float]


def parse_logo_boxes(page: PageRecord) -> list[tuple[Box, float]]:
    """Parse yolo_coords.txt -> [((x1,y1,x2,y2), confidence), ...] sorted by confidence."""
    if not page.raw_dir:
        return []
    f = Path(page.raw_dir) / "yolo_coords.txt"
    if not f.exists():
        return []
    boxes: list[tuple[Box, float]] = []
    for line in f.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        coord_part, _, conf_part = line.partition("\t")
        nums = coord_part.strip().strip("()").split(",")
        try:
            x1, y1, x2, y2 = (float(n) for n in nums[:4])
            conf = float(conf_part) if conf_part.strip() else 0.0
        except ValueError:
            continue
        boxes.append(((x1, y1, x2, y2), conf))
    boxes.sort(key=lambda b: b[1], reverse=True)
    return boxes


def crop_logo(page: PageRecord, pad: int = 4) -> Image.Image | None:
    """Crop the highest-confidence logo box from the screenshot, or None."""
    if not page.screenshot_path or not Path(page.screenshot_path).exists():
        return None
    boxes = parse_logo_boxes(page)
    if not boxes:
        return None
    (x1, y1, x2, y2), _ = boxes[0]
    img = Image.open(page.screenshot_path).convert("RGB")
    w, h = img.size
    left, top = max(0, int(x1) - pad), max(0, int(y1) - pad)
    right, bottom = min(w, int(x2) + pad), min(h, int(y2) + pad)
    if right <= left or bottom <= top:
        return None
    return img.crop((left, top, right, bottom))


class CLIPLogoEmbedder:
    """CLIP zero-shot logo<->brand matcher. Loads the model on first use (CPU OK)."""

    # 'ViT-B-32-quickgelu' is the variant that matches the 'openai' pretrained weights
    # (QuickGELU activation); using plain 'ViT-B-32' triggers an activation mismatch that
    # flattens the similarity margins (~0.02-0.05 observed).
    def __init__(self, model_name: str = "ViT-B-32-quickgelu",
                 pretrained: str = "openai") -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import open_clip  # lazy: requires `pip install -e '.[logo]'`
        import torch  # noqa: F401

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def similarity(self, crop: Image.Image, brand: str) -> float:
        """P(logo matches the claimed brand) via CLIP contrastive softmax in [0,1].

        Raw cosine margins between a logo crop and brand vs generic text are tiny (~0.03);
        applying CLIP's learned logit scale + softmax over [brand, other-brand, generic]
        sharpens them into a usable probability.
        """
        self._ensure()
        import torch

        with torch.no_grad():
            img = self._preprocess(crop).unsqueeze(0)
            prompts = [f"a logo of {brand}", "a logo of a different company", "a screenshot"]
            text = self._tokenizer(prompts)
            img_feat = self._model.encode_image(img)
            txt_feat = self._model.encode_text(text)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            scale = self._model.logit_scale.exp()
            logits = (scale * img_feat @ txt_feat.T).softmax(dim=-1)
            return float(logits[0, 0].item())   # probability of the claimed-brand prompt


def verify_logo_brand(
    cue: Cue, page: PageRecord, embedder: CLIPLogoEmbedder | None = None
) -> float | None:
    """Similarity in [0,1] between the rendered logo and the claimed brand. None if no logo."""
    crop = crop_logo(page)
    if crop is None:
        return None
    if embedder is None:
        try:
            embedder = _default_embedder()
        except Exception:  # noqa: BLE001
            return None
    try:
        return embedder.similarity(crop, cue.value)
    except ImportError:
        # torch / open-clip not installed -> treat logo cue as N/A (dropped from G).
        return None


_DEFAULT: CLIPLogoEmbedder | None = None


def _default_embedder() -> CLIPLogoEmbedder:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CLIPLogoEmbedder()
    return _DEFAULT
