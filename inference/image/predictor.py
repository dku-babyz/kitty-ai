from __future__ import annotations
from typing import Sequence
from pathlib import Path
from PIL import Image
import torch, time
from .model_loader import load_image_model
from ..base import get_device

class ImagePredictor:
    """return 예시
    {
      'safe?': 'Unsafe',
      'safe_prob': 0.02,
      'unsafe_label': 'Sexual',
      'unsafe_prob': 0.91
    }
    """

    def __init__(self, ckpt_path: str | Path):
        self.model, self.id2label, self.tf = load_image_model(ckpt_path)
        self.device = get_device()

    @torch.inference_mode()
    def __call__(self, images: str | Image.Image | Sequence[str | Image.Image]):
        if isinstance(images, (str, Image.Image)):
            images = [images]

        batch = []
        for im in images:
            if isinstance(im, (str, Path)):
                im = Image.open(im).convert("RGB")
            batch.append(self.tf(im))

        x = torch.stack(batch).to(self.device)
        if next(self.model.parameters()).dtype == torch.float16:
            x = x.half()

        tic = time.perf_counter()
        logits = self.model(x); toc = (time.perf_counter() - tic) * 1e3

        probs = logits.softmax(dim=1).cpu()
        safe_idx = next(i for i, v in self.id2label.items() if v == "Safe")

        outputs = []
        for vec in probs:
            safe_p   = float(vec[safe_idx])
            unsafe_idx = int(vec.argmax())
            outputs.append({
                "safe?"       : "Safe" if unsafe_idx == safe_idx else "Unsafe",
                "safe_prob"   : safe_p,
                "unsafe_label": self.id2label[unsafe_idx],
                "unsafe_prob" : float(vec[unsafe_idx]),
            })

        return (outputs[0] if len(outputs) == 1 else outputs), toc  # ms
