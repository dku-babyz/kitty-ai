import pandas as pd
import numpy as np

# 유해 범주 정의
CATEGORIES = ["abuse", "censure", "discrimination", "hate", "sexual", "violence"]

# STEP 1: replace_dataset.csv 불러오기
df = pd.read_csv("replace_dataset.csv")

# STEP 2: 6개 유해성 컬럼을 랜덤 값으로 채우기 (0~1 사이 float)
for cat in CATEGORIES:
    df[cat] = np.random.rand(len(df))  # 혹은 np.random.randint(0, 2, size=len(df)) 로 binary

# STEP 3: temp.csv 저장
df.to_csv("temp.csv", index=False)

# STEP 4: 청크 단위 평균 계산 → site_db.csv 생성
chunk_size = 100
results = []

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    avg_vector = chunk[CATEGORIES].mean().round(4).to_dict()
    dominant = max(avg_vector, key=avg_vector.get)

    result = {
        "site_id": f"site_{i//chunk_size + 1}",
        **avg_vector,
        "dominant_category": dominant
    }
    results.append(result)

result_df = pd.DataFrame(results)
result_df.to_csv("site_db.csv", index=False)
