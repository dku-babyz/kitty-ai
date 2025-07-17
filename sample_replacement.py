# src/sample_replacement.py

from replace import replace_text

# 테스트할 문장 목록
test_sentences = [
    "야이 개쌔끼야",
    "남자 아이돌들은 뭐 하나 같이 호빠 나왔을 거 같냐 언니들 한테 많이 넣어줬겠다",
    "뭐 이런 병신 같은 소리를 하고 있어",
    "저년 진짜 꼴보기 싫다",
    "그냥 기분 나빴어"
]

print("==== 유해 표현 대체 테스트 ====\n")

for i, sent in enumerate(test_sentences, 1):
    print(f"[{i}] 원문:\n{sent}")
    replaced = replace_text(sent, show_prompt=True)  # 디버깅용으로 프롬프트 확인
    print(f"    👉 대체 결과:\n{replaced}")
    print("-" * 60)
