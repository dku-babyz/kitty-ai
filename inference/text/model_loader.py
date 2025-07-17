"""
Utilities to recreate the *exact* model architecture used during training
and load the fine-tuned weights that were saved with 🤗 Transformers'
`Trainer.save_model` (typically `pytorch_model.bin` or `model.safetensors`).
"""
from __future__ import annotations

from pathlib import Path
import torch, torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import (
    AutoModel, AutoTokenizer, RobertaConfig,
    PreTrainedModel, logging,
)
from ..base import get_device

logging.set_verbosity_error()

# ───────────────────────── 1. Model definition ─────────────────────────
class RoBERTaMultiTask(PreTrainedModel):
    config_class = RobertaConfig

    def __init__(
        self,
        config: RobertaConfig,
        *,
        num_labels: int = 6,
        intensity_classes: int = 4,
    ):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config, add_pooling_layer=False)
        h = config.hidden_size
        self.head_label = nn.Linear(h, num_labels)
        # ⚠️  학습 코드와 동일한 이름 유지!
        self.head_inten = nn.Linear(h, intensity_classes)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **_):
        feat = self.backbone(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return self.head_label(feat), self.head_inten(feat)


# ───────────────────────── 2. Loader helpers ───────────────────────────
_WEIGHT_CANDIDATES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.pt",
)

def _find_weight_file(model_dir: Path) -> Path:
    for n in _WEIGHT_CANDIDATES:
        f = model_dir / n
        if f.exists():
            return f
    raise FileNotFoundError(
        f"No weight file found in {model_dir} "
        f"(expected one of: {', '.join(_WEIGHT_CANDIDATES)})"
    )

def _load_state_dict(p: Path) -> dict:
    if p.suffix == ".safetensors":
        return safe_load(str(p))
    return torch.load(str(p), map_location="cpu", weights_only=False)


# ───────────────────────── 3. Public API ───────────────────────────────
def load_text_model(model_dir: str | Path | None = None):
    """
    If `model_dir` is None → `inference/text/model/` 를 기본값으로.
    """
    model_dir = (
        Path(model_dir)
        if model_dir is not None
        else Path(__file__).resolve().parent / "model"
    )
    if not model_dir.exists():
        raise FileNotFoundError(
            f"'{model_dir}' 경로가 없습니다. 학습 완료 모델을 넣어 주세요."
        )

    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = RobertaConfig.from_pretrained(model_dir)

    model = RoBERTaMultiTask(cfg)
    sd = _load_state_dict(_find_weight_file(model_dir))
    model.load_state_dict(sd, strict=True)
    model.eval().to(get_device())

    return tok, model
