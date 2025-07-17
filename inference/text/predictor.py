from __future__ import annotations
import torch, numpy as np
from typing import Sequence, List, Dict, Any
from .model_loader import load_text_model
from ..base import get_device

__all__ = ["TextPredictor"]

LABEL_COLUMNS: List[str] = ["abuse", "censure", "hate", "discrimination", "sexual", "violence"]
INTENSITY_MAP = {0: "none", 1: "mild", 2: "medium", 3: "severe"}

class TextPredictor:
    """Thin wrapper around the fine‑tuned RoBERTa multitask model.

    Examples
    --------
    >>> predictor = TextPredictor("./inference/text/model")
    >>> predictor("너 정말 못됐어")
    {'labels': ['abuse'], 'probs': [0.9123], 'intensity': 'mild'}
    """

    def __init__(self, model_dir: str | os.PathLike):
        self.tokenizer, self.model = load_text_model(model_dir)
        self.device = get_device()

    @torch.inference_mode()
    def __call__(self,
                 texts: str | Sequence[str],
                 *,
                 threshold: float = 0.5,
                 return_logits: bool = False
                 ) -> Dict[str, Any] | List[Dict[str, Any]]:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        enc = self.tokenizer(list(texts),
                             padding=True,
                             truncation=True,
                             max_length=128,
                             return_tensors="pt").to(self.device)

        logit_labels, logit_intensity = self.model(**enc)  # (B, 6) ‖ (B, 4)

        # ─────── labels ───────
        probs = torch.sigmoid(logit_labels).cpu().numpy()  # (B, 6)
        intens_cls = logit_intensity.softmax(dim=-1).argmax(dim=-1).cpu().tolist()

        outputs = []
        for p_vec, inten_idx in zip(probs, intens_cls):
            chosen = np.where(p_vec >= threshold)[0]
            item = {
                "labels": [LABEL_COLUMNS[i] for i in chosen.tolist()],
                "probs":  [float(p_vec[i]) for i in chosen.tolist()],
                "intensity": INTENSITY_MAP[inten_idx]
            }
            if return_logits:
                item["logits_label"] = p_vec.tolist()
                # raw intensity logits
            outputs.append(item)

        return outputs[0] if single_input else outputs