import os
import sys
import pandas as pd
from tqdm import tqdm

# src 디렉터리를 import 경로에 추가
sys.path.append(os.path.abspath("."))

from inference.text import TextPredictor
from inference.dictionary import DictionaryChecker
from replace import replace_text, should_filter
import re
import ast

def parse_gpt_output(raw_output: str) -> dict:
    """
    GPT 응답에서 JSON-like 딕셔너리 추출 시도
    """
    # ```json 또는 ``` 제거
    cleaned = re.sub(r"```(?:json)?", "", raw_output).strip("` \n")

    # 중괄호 블록만 추출
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}

    try:
        return ast.literal_eval(match.group())
    except Exception:
        return {}


# CSV 불러오기
df = pd.read_csv("replace_dataset.csv").head(100)  # 예시로 1000개만 사용

# 모델 초기화
checker = DictionaryChecker("inference/dictionary/dictionary.csv")
txt_pred = TextPredictor("inference/text/model")

# 결과 컬럼 초기화
df["사전_유해성"] = 0
df["AI_유해성"] = 0
df["유해_단어"] = ""
df["대체_제안형식"] = ""
df["대체_문장"] = ""

# 진행도 표시
for idx, row in tqdm(df.iterrows(), total=len(df), desc="🚀 Processing"):
    text = row["text"]
    dict_result = checker(text)
    ai_result = txt_pred(text)

    dict_flag = 1 if dict_result else 0
    ai_flag = 1 if ai_result.get("labels") else 0

    df.at[idx, "사전_유해성"] = dict_flag
    df.at[idx, "AI_유해성"] = ai_flag

    if not should_filter(dict_result, ai_result):
        continue

    try:
        replaced = replace_text(text, show_prompt=False)
        result_json = parse_gpt_output(replaced)

        if result_json:
            df.at[idx, "유해_단어"] = result_json.get("문장중 유해한 단어들", "")
            df.at[idx, "대체_제안형식"] = result_json.get("대체 제안 형식", "")
            df.at[idx, "대체_문장"] = result_json.get("대체 문장", "")
        else:
            df.at[idx, "대체_문장"] = replaced  # 일반 텍스트일 경우

    except Exception as e:
        df.at[idx, "대체_문장"] = f"Error: {str(e)}"

# 🔍 평가
y_true = df["intensity"]
y_pred = (df["사전_유해성"] | df["AI_유해성"]).astype(int)



# 🔽 CSV 저장
df.to_csv("replace_dataset_output.csv", index=False, encoding="utf-8-sig")
print("\n✅ 결과 저장 완료: replace_dataset_output.csv")
