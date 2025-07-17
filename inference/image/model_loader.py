# inference/image/model_loader.py  ⛔ timm · lightning 제거 버전
from __future__ import annotations
from pathlib import Path
import json, torch
from torchvision import transforms
from ..base import get_device

# ─── 1) 이미지 전처리 ──────────────────────────────────────────────────
IMG_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# ─── 2) TorchScript 로더 ─────────────────────────────────────────────
def load_image_model(ts_path: str | Path):
    """
    Parameters
    ----------
    ts_path : str | Path
        TorchScript 파일 (.ts.pt)  ← Colab에서 변환한 결과물
        옆에 같은 이름의 .json    ← label2id 매핑

    Returns
    -------
    model      : torch.jit.ScriptModule (eval & to(device))
    id2label   : dict[int,str]
    transform  : torchvision.transforms
    """
    ts_path = Path(ts_path)
    device  = get_device()

    # (1) TorchScript 모델
    model = torch.jit.load(str(ts_path), map_location=device)
    if device.type == "cuda":
        model.half()                  # fp16 → VRAM ↓
    model.eval()

    # (2) 레이블 매핑
    jpath      = ts_path.with_suffix(".json")
    id2label   = {int(v): k for k, v in json.loads(jpath.read_text()).items()}

    return model, id2label, IMG_TF
