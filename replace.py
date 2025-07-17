import os
from inference.text import TextPredictor
from inference.dictionary import DictionaryChecker
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 모델 초기화
checker = DictionaryChecker("inference/dictionary/dictionary.csv")
txt_pred = TextPredictor("inference/text/model")

# 템플릿 로드
def load_instruction_template(path="prompts/replace_template.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_dict_info(result_dict):
    summaries = []
    for word, info in result_dict.items():
        summaries.append(f"{word}({info.get('분류')}, {info.get('유형')}, 의미: {info.get('의미')})")
    return "\n".join(f"- {s}" for s in summaries)

def should_filter(dict_result, ai_result):
    return bool(dict_result) or bool(ai_result.get("labels"))

def build_prompt(original, dict_summary, ai_result, extra_instruction):
    prompt = f"""다음 문장은 유해할 수 있는 표현을 포함하고 있습니다:

원문: "{original}"

"""
    if dict_summary:
        prompt += f"사전에서 감지된 유해 단어:\n{dict_summary}\n"

    if ai_result.get("labels"):
        prompt += f"\nAI 모델 분석 결과:\n라벨: {', '.join(ai_result['labels'])}, 강도: {ai_result['intensity']}\n"

    prompt += f"\n{extra_instruction}"
    return prompt

def replace_text(text, model="gpt-4o-mini", show_prompt=False):
    dict_result = checker(text)
    ai_result = txt_pred(text)

    if not should_filter(dict_result, ai_result):
        return text  # 유해하지 않으면 원본 그대로

    dict_summary = extract_dict_info(dict_result) if dict_result else ""
    extra_instruction = load_instruction_template()

    final_prompt = build_prompt(text, dict_summary, ai_result, extra_instruction)

    if show_prompt:
        print("\n===== [GPT에 전달될 프롬프트] =====\n")
        print(final_prompt)
        print("\n===== [END PROMPT] =====\n")

    response = client.responses.create(
        model=model,
        input=final_prompt
    )
    return response.output_text.strip()

# 테스트
if __name__ == "__main__":
    test_text = "유해텍스트"
    print("원문:", test_text)
    print("대체 결과:", replace_text(test_text))
