# 📚 Safe/Unsafe Inference Kit

혼합 한국어/영어로 설명된 유해 추론 키트입니다.

## 📂️ 디렉터리 구조

```bash
project/
│
├─ inference/                ← 파이썬 패키지 (text / image / dictionary)
│   ├─ base.py
│   ├─ text/
│   ├─ image/
│   └─ dictionary/
│
├─ inference/image/model_ts.pt     ← Colab 에서 만든 TorchScript 메뉴
├─ inference/image/model_ts.json  ← label2id 매칭 (uc790동 저장)
│
├─ inference/text/model/          ← HuggingFace Trainer.save_model 결과물
│   ├─ config.json, pytorch_model.bin ...
│
├─ inference/dictionary/사전.csv
│
├─ tmp.py            ← 로컬 벿치마크용 샘플 스크립트
└─ env.py            ← GPU / PyTorch 확인용 스크립트
```

---

## ✅ 1. DictionaryChecker

```python
from inference.dictionary import DictionaryChecker

chk = DictionaryChecker("inference/dictionary/사전.csv")
result = chk("야 (개삭녀이) 무하냐?")
print(result)
```

### 🔄 반환 형식

```python
{
  "<matched_word>": {
    "언어표현": "...",
    "분류": "...",
    "유형": "...",
    "품사": "...",
    "의미": "...",
    "예문": "...",
    "비고": "..."
  },
  ...
}
```

- **Aho–Corasick Trie 탐색** 사용
- 4,500 단어 기준 < `0.1ms` 처리
- 괴호, 대소문자, 전각/반각 → 자동 정규화 (`NFKC`)

---

## 🧠 2. TextPredictor

```python
from inference.text import TextPredictor

txt_pred = TextPredictor("inference/text/model")
out = txt_pred("무약테스트", threshold=0.5)
print(out)
```

### ⚙️ 입력 인자

| 파라미터       | 타입              | 기본값   | 설명                             |
|----------------|-------------------|----------|----------------------------------|
| `texts`        | str / list[str]   | –        | 단일 문장 또는 문장 리스트        |
| `threshold`    | float             | 0.5      | 확률 ≥ threshold ➞ 탐지          |
| `return_logits`| bool              | False    | True 시 logit 벡터 포�          |

### 🔄 반환 예시

```python
{
  'labels'   : ['abuse', 'sexual'],
  'probs'    : [0.82, 0.74],
  'intensity': 'medium'
  # return_logits=True 시 → 'logits_label': [...]
}
```

- **batch-safe**: 리스트 입력 → 리스트 출력
- 추론 속도 (GPU fp16) ≈ `25 ms` / 문장

---

## 🖼️ 3. ImagePredictor

```python
from inference.image import ImagePredictor

img_pred = ImagePredictor("inference/image/model_ts.pt")
pred, latency_ms = img_pred("example.jpg")
print(pred, latency_ms)
```

### 📅 입력 타입

- 파일 경로 문자열 (`"cat.jpg"`)
- `PIL.Image.Image`
- 또는 두 가지 타입을 조합한 list

### 🔄 반환

- 단일 이미지 → `(dict, latency_ms)`
- 배치 → `(list[dict], latency_ms)`

```python
{
  'safe?'       : 'Unsafe',
  'safe_prob'   : 0.0014,
  'unsafe_label': 'Sexual',
  'unsafe_prob' : 0.9985
}
```

- TorchScript 모델 로드 ≈ `0.6 s`, 추론 ≈ `30 ms` / 장
- `model.half()` 적용 시 VRAM 사용량 ≈ `3 GB`

---

## ⚡ 4. 빠른 벿치마크 (`tmp.py`)

```python
from inference.dictionary import DictionaryChecker
from inference.text        import TextPredictor
from inference.image       import ImagePredictor
import time

chk  = DictionaryChecker("inference/dictionary/사전.csv")
txt  = TextPredictor("inference/text/model")
img  = ImagePredictor("inference/image/model_ts.pt")

def bench():
    t0=time.time(); chk("야 (개삭녀이) 무하냐");  print("dict", (time.time()-t0)*1e3,"ms")
    t0=time.time(); txt("유해 테스트");          print("text", (time.time()-t0)*1e3,"ms")
    t0=time.time(); img("example.jpg");         print("img ", (time.time()-t0)*1e3,"ms")

bench()
```

---

## 📦 Requirements

```txt
torch>=2.3.0
torchvision>=0.18.0
transformers
safetensors

ahocorasick
pandas
```

---