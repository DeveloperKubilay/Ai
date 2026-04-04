import requests, json

config = json.load(open("settings.json", "r", encoding="utf-8"))
TEACHER_PROMPT = open("teacher_prompt.md", "r", encoding="utf-8").read()

def process_content(content):
    prompt = TEACHER_PROMPT.format(num_questions=config["training"]["num_questions"], content=content[:2000])
    payload = {"model": config["teacher"]["model"], "prompt": prompt, "stream": False, 
               "options": {"temperature": 0.7, "num_predict": 4000}}
    
    for attempt in range(3):
        try:
            result = requests.post(config["teacher"]["url"], json=payload, timeout=240).json().get("response", "")
            start, end = result.find('['), result.rfind(']') + 1
            if start == -1 or end <= start: raise ValueError("JSON bulunamadı")
            
            qa_list = json.loads(result[start:end].replace('\n', ' ').replace('\r', ' '))
            if not isinstance(qa_list, list) or len(qa_list) == 0: raise ValueError("Geçersiz liste")
            
            print(f"  ✅ {len(qa_list)} soru-cevap")
            return [{"text": f"<|im_start|>user\n{qa.get('soru', '')}<|im_end|>\n<|im_start|>assistant\n{qa.get('cevap', '')}<|im_end|>"} 
                    for qa in qa_list if qa.get('soru') and qa.get('cevap')]
        except Exception as e:
            print(f"  ⚠️ Hata ({attempt+1}/3): {e}")
            if attempt == 2: return []
    return []

print("📥 input.jsonl işleniyor...")
train_file = open("train.jsonl", "w", encoding="utf-8")
total = 0

for line_num, line in enumerate(open("input.jsonl", "r", encoding="utf-8"), 1):
    item = json.loads(line.strip())
    print(f"[{line_num}] İşleniyor...")
    
    if "url" in item:
        response = requests.get(item["url"], timeout=10)
        try:
            data = response.json()
            content = f"{data.get('description', '')}\n\n{data.get('readme', '')[:2000]}" if "readme" in data else json.dumps(data, ensure_ascii=False)[:2000]
        except:
            content = response.text[:2000]
    elif "text" in item:
        content = item["text"][:2000]
    else:
        print("  ⚠️ Geçersiz format")
        continue
    
    formatted_data = process_content(content)
    for data in formatted_data:
        train_file.write(json.dumps(data, ensure_ascii=False) + "\n")
    total += len(formatted_data)

train_file.close()
print(f"✅ Toplam {total} örnek → python index.py")
