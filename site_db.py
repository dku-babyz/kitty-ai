import pandas as pd

# 유해 범주 정의
CATEGORIES = ["abuse", "censure", "discrimination", "hate", "sexual", "violence"]

# CSV 불러오기
df = pd.read_csv("replace_dataset.csv")

# 청크 단위로 나눠서 평균 계산
chunk_size = 100
results = []

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    avg_vector = chunk[CATEGORIES].mean().round(4).to_dict()
    
    dominant = max(avg_vector, key=avg_vector.get)
    
    result = {
        **avg_vector
    }
    
    results.append(result)

# 결과 저장
result_df = pd.DataFrame(results)
result_df.to_csv("site_db.csv", index=False)
