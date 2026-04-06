import json

import requests

from pipeline_utils import load_config, project_path


config = load_config()
teacher_prompt_path = project_path("docs", "teacher_prompt.md")
input_path = project_path("data", "input.jsonl")
train_path = project_path("data", "train.jsonl")

with open(teacher_prompt_path, "r", encoding="utf-8") as f:
    teacher_prompt = f.read()


def process_content(content: str) -> list[dict]:
    prompt = teacher_prompt.format(
        num_questions=config["training"]["num_questions"],
        content=content[:2000],
    )
    payload = {
        "model": config["teacher"]["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 4000,
        },
    }

    print("  Teacher'a gonderiliyor...")
    for attempt in range(3):
        try:
            response = requests.post(config["teacher"]["url"], json=payload, timeout=240)
            result = response.json().get("response", "")

            start = result.find("[")
            end = result.rfind("]") + 1
            if start == -1 or end <= start:
                raise ValueError("JSON array bulunamadi")

            qa_list = json.loads(result[start:end].replace("\n", " ").replace("\r", " "))
            if not isinstance(qa_list, list) or not qa_list:
                raise ValueError("Gecersiz QA listesi")

            print(f"  {len(qa_list)} soru-cevap uretildi")
            formatted_data = []
            for qa in qa_list:
                question = qa.get("soru", "")
                answer = qa.get("cevap", "")
                if question and answer:
                    text = (
                        f"<|im_start|>user\n{question}<|im_end|>\n"
                        f"<|im_start|>assistant\n{answer}<|im_end|>"
                    )
                    formatted_data.append({"text": text})
            return formatted_data
        except Exception as exc:
            print(f"  Hata ({attempt + 1}/3): {exc}")
            if attempt == 2:
                print("  Teacher cevap veremedi, atlaniyor")
                return []
            print("  Tekrar deneniyor...")
    return []


print("=" * 60)
print("INPUT.JSONL ISLENIYOR")
print("=" * 60)

total_processed = 0
with open(train_path, "w", encoding="utf-8") as train_file:
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            item = json.loads(line.strip())
            print(f"\n[{line_num}] Isleniyor...")

            if "url" in item:
                print(f"  URL: {item['url']}")
                response = requests.get(item["url"], timeout=10)
                try:
                    data = response.json()
                    if "readme" in data:
                        content = f"{data.get('description', '')}\n\n{data.get('readme', '')[:2000]}"
                    else:
                        content = json.dumps(data, ensure_ascii=False)[:2000]
                except Exception:
                    content = response.text[:2000]
            elif "text" in item:
                print(f"  Text: {len(item['text'])} karakter")
                content = item["text"][:2000]
            else:
                print("  Gecersiz format, atlaniyor")
                continue

            formatted_data = process_content(content)
            for data in formatted_data:
                train_file.write(json.dumps(data, ensure_ascii=False) + "\n")

            total_processed += len(formatted_data)
            print(f"  {len(formatted_data)} ornek data/train.jsonl'e eklendi")

print("\n" + "=" * 60)
print(f"TAMAMLANDI! Toplam {total_processed} ornek")
print("=" * 60)
print("Siradaki adimlar:")
print("python scripts/prepare_dataset.py")
print("python scripts/index.py")
