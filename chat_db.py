import pandas as pd

# STEP 1: 유해성 범주 정의
CATEGORIES = ["abuse", "censure", "discrimination", "hate", "sexual", "violence"]

# CSV 불러오기
df = pd.read_csv("replace_dataset.csv")

# STEP 2: 평균 벡터 계산
avg_vector = df[CATEGORIES].mean().round(4).to_dict()

# dominant category = 가장 평균값이 높은 항목
dominant_category = max(avg_vector, key=avg_vector.get)

# 결과 기록용 dict
result = {
    **avg_vector
}

# STEP 3: DB로 저장
out_df = pd.DataFrame([result])
out_df.to_csv("chat_db.csv", index=False)
