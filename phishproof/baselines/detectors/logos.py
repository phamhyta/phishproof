"""Thin adapter over Phishpedia's released logo detector and Siamese model, and a nearest-logo index.

Serves the logo brand extractor of KnowPhish Sec. 5.3 and the Logo Extractor of PhishAgent Sec. 4.2 and App. C.
KnowPhish reuses PhishIntention's modules (App. A.1); this re-implementation uses Phishpedia's for both detectors.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass
class PhishpediaConfig:
    """Import paths and model files of a Phishpedia checkout."""

    repo: str
    detector_module: str = "logo_recog"
    detector_config_fn: str = "config_rcnn"
    detector_predict_fn: str = "pred_rcnn"
    matcher_module: str = "logo_matching"
    siamese_load_fn: str = "load_model_weights"
    siamese_embed_fn: str = "get_embedding"
    rcnn_config: str = "models/faster_rcnn.yaml"
    rcnn_weights: str = "models/rcnn_bet365.pth"
    # a detector score floor of 0.05, the value recalled from Phishpedia's configs.yaml.
    detect_threshold: float = 0.05
    siamese_weights: str = "models/resnetv2_rgb_new.pth.tar"
    siamese_num_classes: int = 277

    @classmethod
    def load(cls, repo: str, overrides_path: str | None = None) -> PhishpediaConfig:
        overrides: dict[str, Any] = {}
        if overrides_path:
            overrides = json.loads(Path(overrides_path).read_text(encoding="utf-8"))
        return cls(**{"repo": repo, **overrides})


class LogoBackend(Protocol):
    def top_logo(self, screenshot_path: str | Path) -> Image.Image | None: ...

    def embed(self, logo: Image.Image) -> np.ndarray: ...


class PhishpediaAdapter:
    """Run Phishpedia's detector and Siamese embedding; everything else in this package goes through this class."""

    def __init__(self, config: PhishpediaConfig) -> None:
        self.config = config
        self.repo = Path(config.repo).expanduser().resolve()
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        self.verify_api()
        detector_module = importlib.import_module(config.detector_module)
        matcher_module = importlib.import_module(config.matcher_module)
        self._predict = getattr(detector_module, config.detector_predict_fn)
        self._embed = getattr(matcher_module, config.siamese_embed_fn)
        make_detector = getattr(detector_module, config.detector_config_fn)
        load_siamese = getattr(matcher_module, config.siamese_load_fn)
        self._detector = make_detector(
            str(self.repo / config.rcnn_config), str(self.repo / config.rcnn_weights), config.detect_threshold
        )
        self._siamese = load_siamese(config.siamese_num_classes, str(self.repo / config.siamese_weights))
        self._lock = threading.Lock()

    def verify_api(self) -> None:
        """Fail with a clear message when the checkout does not expose the expected functions."""
        expected = {
            self.config.detector_module: [self.config.detector_config_fn, self.config.detector_predict_fn],
            self.config.matcher_module: [self.config.siamese_load_fn, self.config.siamese_embed_fn],
        }
        problems: list[str] = []
        for module_name, functions in expected.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                problems.append(f"cannot import {module_name}: {exc}")
                continue
            problems.extend(f"{module_name}.{name} not found" for name in functions if not hasattr(module, name))
        if problems:
            raise RuntimeError(
                "Phishpedia checkout does not match the adapter: " + "; ".join(problems)
            )

    def top_logo(self, screenshot_path: str | Path) -> Image.Image | None:
        """Return the crop of the highest-scoring logo box, or None when no logo is detected."""
        with self._lock:
            boxes = self._predict(str(screenshot_path), self._detector)
        if boxes is None or len(boxes) == 0:
            return None
        # boxes arrive in descending score order, as Detectron2 returns them, so the first is the top logo (PhishAgent App. C ranks by score and takes the top one).
        left, top, right, bottom = (int(value) for value in boxes[0].tolist())
        with Image.open(screenshot_path) as image:
            rgb = image.convert("RGB")
        box = (max(left, 0), max(top, 0), min(right, rgb.width), min(bottom, rgb.height))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return rgb.crop(box)

    def embed(self, logo: Image.Image) -> np.ndarray:
        with self._lock:
            vector = self._embed(logo, self._siamese)
        return _unit(np.asarray(vector, dtype=np.float32))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


class LogoIndex:
    """Reference logo embeddings of the knowledge base."""

    def __init__(self, embeddings: np.ndarray, brand_ids: list[str]) -> None:
        if len(embeddings) != len(brand_ids):
            raise ValueError("embeddings and brand_ids differ in length")
        self.embeddings = embeddings
        self.brand_ids = brand_ids

    @classmethod
    def load(cls, path: str | Path) -> LogoIndex:
        with np.load(path, allow_pickle=False) as data:
            return cls(data["embeddings"], [str(value) for value in data["brand_ids"].tolist()])

    def save(self, path: str | Path) -> None:
        np.savez(path, embeddings=self.embeddings, brand_ids=np.array(self.brand_ids, dtype=str))

    def best(self, embedding: np.ndarray) -> tuple[str, float] | None:
        """Return the brand of the nearest reference logo and its cosine similarity."""
        if len(self.brand_ids) == 0:
            return None
        scores = self.embeddings @ _unit(embedding)
        # one logo matches one brand by top-1 similarity, with no resolution-alignment retry, because the paper says only that the logo is matched to a brand.
        index = int(np.argmax(scores))
        return self.brand_ids[index], float(scores[index])
