# infer.py  ───────────────────────────────────────────────────────────────
import torch, json
import numpy as np
from scipy.special import expit, softmax
from transformers import (AutoTokenizer, AutoModel, PreTrainedModel,
                          RobertaConfig)
import torch.nn as nn

# ─────────────────── 0. 사용자 설정 ───────────────────
MODEL_DIR      = "./model"          # <<< 그림의 폴더 이름
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN        = 128                # 학습에 썼던 hparams["max_length"]

label_cols        = ["abuse", "censure", "hate",
                     "discrimination", "sexual", "violence"]
intensity_classes = ["none", "low", "mid", "high"]

# ─────────────────── 1. 모델 정의 (학습과 동일) ───────────────────
class RoBERTaMultiTask(PreTrainedModel):
    config_class = RobertaConfig
    def __init__(self, config, num_labels: int):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config._name_or_path, add_pooling_layer=False)
        hid = config.hidden_size
        self.head_label = nn.Linear(hid, num_labels)          # 6‑label
        self.head_inten = nn.Linear(hid, 4)                   # 4‑class intensity
        self.sigmoid = nn.Sigmoid()
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kw):
        feat = self.backbone(input_ids, attention_mask=attention_mask
                             ).last_hidden_state[:, 0]        # [CLS]
        logits_lbl = self.head_label(feat)                    # (B,6)
        logits_int = self.head_inten(feat).squeeze(-1)        # (B,4)
        return (logits_lbl, logits_int)

# ─────────────────── 2. 로드 ───────────────────
cfg   = RobertaConfig.from_pretrained(MODEL_DIR)
model = RoBERTaMultiTask.from_pretrained(MODEL_DIR,
                                         config=cfg,
                                         num_labels=len(label_cols))
model.to(DEVICE).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# ─────────────────── 3. 추론 함수 ───────────────────
@torch.inference_mode()
def predict(texts):
    """
    texts: str | list[str] | dict[key, str]
    return: list[dict]  (입력 순서대로)
    """
    if isinstance(texts, str):
        items = [(0, texts)]
    elif isinstance(texts, list):
        items = list(enumerate(texts))
    elif isinstance(texts, dict):
        items = list(texts.items())
    else:
        raise TypeError("Input must be str / list[str] / dict[str, str].")

    results = []
    for key, sentence in items:
        enc = tokenizer(sentence,
                        padding="max_length", truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)

        logits_lbl, logits_int = model(**enc)          # forward()
        probs_lbl = expit(logits_lbl.cpu().numpy().squeeze())      # (6,)
        probs_int = softmax(logits_int.cpu().numpy().squeeze())    # (4,)

        results.append({
            "id"          : key,
            "text"        : sentence,
            "intensity"   : intensity_classes[int(probs_int.argmax())],
            "intensity_probs": {c: float(p) for c, p in zip(intensity_classes, probs_int)},
            "label_probs" : {c: float(p) for c, p in zip(label_cols, probs_lbl)}
        })

    return results

# ─────────────────── 4. 예시 실행 ───────────────────
if __name__ == "__main__":
    sample_sentences = [
        "이 사람은 정말 못됐어!",
        "평범한 일상 대화였습니다."
    ]
    out = predict(sample_sentences)

    # 보기 좋게 출력
    print(json.dumps(out, indent=2, ensure_ascii=False))
