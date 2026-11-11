"""Logo check: crop the page logo and score it against the claimed brand with CLIP.

The score is continuous and is recorded for the audit trail. It does not gate an action.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..schema import Cue, PageRecord

Box = tuple[float, float, float, float]


def parse_logo_boxes(page: PageRecord) -> list[tuple[Box, float]]:
    """Read the logo boxes stored with a capture, highest confidence first."""
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
    """Zero-shot logo matcher. The model loads on first use."""

    # The "openai" weights were trained with QuickGELU, so plain ViT-B-32 would not match them.
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
        import open_clip
        import torch  # noqa: F401

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def similarity(self, crop: Image.Image, brand: str) -> float:
        """Softmax probability of the claimed-brand prompt against two alternatives."""
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
            return float(logits[0, 0].item())


def verify_logo_brand(
    cue: Cue, page: PageRecord, embedder: CLIPLogoEmbedder | None = None
) -> float | None:
    """Score of the claimed brand on the page logo, or None when the check is unavailable."""
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
        return None


_DEFAULT: CLIPLogoEmbedder | None = None


def _default_embedder() -> CLIPLogoEmbedder:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CLIPLogoEmbedder()
    return _DEFAULT


def select_logo_threshold(wrong_brand_scores: list[float], max_false_accept: float = 0.05) -> float:
    """Lowest score whose false acceptance on wrong-brand pairs stays within the limit."""
    if not wrong_brand_scores:
        raise ValueError("need at least one wrong-brand score")
    total = len(wrong_brand_scores)
    for candidate in sorted(set(wrong_brand_scores)):
        accepted = sum(score >= candidate for score in wrong_brand_scores)
        if accepted / total <= max_false_accept:
            return float(candidate)
    return float(max(wrong_brand_scores)) + 1e-9
