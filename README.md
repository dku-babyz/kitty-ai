# 🧠 AI 기반 유해 표현 탐지 및 순화 시스템

이 프로젝트는 텍스트 및 이미지 내의 유해 표현을 탐지하고, GPT API를 활용해 **자연스럽고 정제된 표현**으로 바꾸는 기능을 제공합니다.

---

## 📁 프로젝트 구조

```
src/
├── inference/                  ← 모델 추론 모듈 (텍스트/이미지/사전 기반)
├── prompts/                    ← 프롬프트 템플릿 보관
├── replace.py                  ← 유해 표현 감지 + GPT 순화 처리
├── sample_detection.py         ← 탐지 기능 테스트용 스크립트
├── sample_replacement.py       ← 순화 결과 확인 스크립트
├── requirements.txt
└── README.md
```

---

## 🛠️ 설치 방법

1. Python 3.10 이상 사용 권장
2. 가상환경을 만든 뒤, 아래 명령어로 필수 라이브러리를 설치하세요.

```bash
pip install -r requirements.txt
```

> ⚠️ 주의: `torch` 버전은 CUDA 환경에 따라 달라질 수 있습니다.  
> GPU 사용 시, 예를 들어 다음 명령어를 사용할 수 있습니다:  


---

## 💾 모델 파일 다운로드 및 배치

### ✅ 텍스트 모델 (RoBERTa 기반 다중 분류)

- 다운로드: [txt model](https://drive.google.com/file/d/1lXj6CXd2GbhELbti6nxVSg8-Y609Ea-w/view?usp=sharing)
- 압축 해제 후 → 아래 경로에 저장:

```
inference/text/model/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.json
└── ...
```

---

### ✅ 이미지 모델 (TorchScript 포맷)

- 다운로드: [image model](https://drive.google.com/file/d/1fz5P6DDL4hhWwJJbsxfX_zYu13h5nhFz/view?usp=sharing)
- `.ts.pt` 및 `.json` 파일을 아래 경로에 저장:

```
inference/image/
├── model_ts.pt
├── model_ts.json
```

---

## ⚙️ 사용 방법

### 🔍 유해 표현 탐지 (텍스트/사전 기반)

```bash
python src/sample_detection.py
```

- `inference/dictionary/`에 정의된 유해 단어 기반 감지
- `inference/text/`는 학습된 모델 기반 판단 (ex. abuse, sexual, violence 등)

---

### 🪄 유해 표현 순화 (GPT 기반)

```bash
python src/sample_replacement.py
```

- `replace_text(text)` 함수 호출 시 OpenAI API를 통해 정제된 표현 반환
- `prompts/replace_template.txt`에서 문체/어투 템플릿을 자유롭게 설정 가능

---

## 📦 주요 함수 설명

### 1. `replace_text(text: str, model="gpt-4o-mini", show_prompt=False) -> str`

- 유해 표현이 감지된 경우 GPT로 순화된 문장을 반환
- 아무 문제 없으면 원문 그대로 반환
- `show_prompt=True`로 설정 시, GPT에 전달되는 프롬프트를 확인 가능

```python
from replace import replace_text
replace_text("야이 개쌔끼야", show_prompt=True)
```

---

### 2. `TextPredictor`, `DictionaryChecker` 직접 사용

```python
from inference.text import TextPredictor
from inference.dictionary import DictionaryChecker

txt_pred = TextPredictor("inference/text/model")
checker = DictionaryChecker("inference/dictionary/dictionary.csv")

print(txt_pred("유해 텍스트"))
print(checker("유해 텍스트"))
```

---

## 🧪 테스트 스크립트

- `sample_detection.py` : 유해 표현 탐지 속도/결과 확인
- `sample_replacement.py` : 순화 결과 테스트용

---

## 🔐 OpenAI API Key 설정

GPT 기반 순화 기능을 사용하기 위해 OpenAI API 키를 환경 변수로 설정해야 합니다:

```bash
echo 'export OPENAI_API_KEY=your_openai_key_here' >> ~/.bashrc
source ~/.bashrc
```

또는 Python 코드에서 직접 설정할 수도 있습니다:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_key_here"
```

---

## 🤖 지원 모델

아래 OpenAI GPT 모델들과 호환됩니다:

- `gpt-4o`, `gpt-4o-mini`, `gpt-4.1-mini` 등

응답은 아래 형식 중 하나로 제공됩니다:

- 정제된 단일 문장
- 또는 JSON 형식 결과 (프롬프트 템플릿 설정에 따름)

---
