import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# src 디렉터리를 import 경로에 추가
sys.path.append(os.path.abspath("."))

from inference.text import TextPredictor
from inference.dictionary import DictionaryChecker
from replace import replace_text, should_filter

# CSV 불러오기
df = pd.read_csv("replace_dataset.csv").head(1000)  # 예시로 1000개만 사용

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
        if replaced.startswith("{"):
            result_json = eval(replaced)
            df.at[idx, "유해_단어"] = result_json.get("문장중 유해한 단어들", "")
            df.at[idx, "대체_제안형식"] = result_json.get("대체 제안 형식", "")
            df.at[idx, "대체_문장"] = result_json.get("대체 문장", "")
        else:
            df.at[idx, "대체_문장"] = replaced
    except Exception as e:
        df.at[idx, "대체_문장"] = f"Error: {str(e)}"

# 🔍 평가
y_true = df["intensity"]
y_pred = (df["사전_유해성"] | df["AI_유해성"]).astype(int)

# acc = accuracy_score(y_true, y_pred)
# prec = precision_score(y_true, y_pred, zero_division=0)
# rec = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)

# print("\n🎯 Evaluation Metrics:")
# print(f"Accuracy : {acc:.4f}")
# print(f"Precision: {prec:.4f}")
# print(f"Recall   : {rec:.4f}")
# print(f"F1 Score : {f1:.4f}")

# 🔽 CSV 저장
df.to_csv("replace_dataset_output.csv", index=False, encoding="utf-8-sig")
print("\n✅ 결과 저장 완료: replace_dataset_output.csv")
