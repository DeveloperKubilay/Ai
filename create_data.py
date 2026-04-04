import requests
import json

# ============================================
# YAPILANDIRMA
# ============================================

with open("settings.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open("teacher_prompt.md", "r", encoding="utf-8") as f:
    TEACHER_PROMPT = f.read()

# ============================================
# TEACHER İLE SORU-CEVAP ÜRET VE FORMATLA
# ============================================

def process_content(content):
    """İçerik al, teacher'a sor, formatlanmış veri döndür"""
    
    # Teacher'a gönder
    prompt = TEACHER_PROMPT.format(
        num_questions=config["training"]["num_questions"],
        content=content[:2000]
    )
    
    payload = {
        "model": config["teacher"]["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,  # Daha düşük = daha tutarlı JSON
            "num_predict": 4000
        }
    }
    
    print(f"  🤖 Teacher'a gönderiliyor...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(config["teacher"]["url"], json=payload, timeout=240)
            result = response.json().get("response", "")
            
            # JSON array bul
            start = result.find('[')
            end = result.rfind(']') + 1
            
            if start == -1 or end <= start:
                raise ValueError("JSON array bulunamadı")
            
            json_str = result[start:end]
            
            # JSON'u temizle (yaygın hatalar)
            json_str = json_str.replace('\n', ' ')
            json_str = json_str.replace('\r', ' ')
            
            qa_list = json.loads(json_str)
            
            if not isinstance(qa_list, list) or len(qa_list) == 0:
                raise ValueError("Geçersiz QA listesi")
            
            print(f"  ✅ {len(qa_list)} soru-cevap üretildi")
            
            # Qwen formatına çevir
            formatted_data = []
            for qa in qa_list:
                q = qa.get("soru", "")
                a = qa.get("cevap", "")
                
                if q and a:  # Boş değilse
                    text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
                    formatted_data.append({"text": text})
            
            return formatted_data
            
        except Exception as e:
            print(f"  ⚠️ Hata (deneme {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"  ❌ Teacher cevap veremedi, atlanıyor")
                return []
            print(f"  🔄 Tekrar deneniyor...")
    
    return []

# ============================================
# INPUT.JSONL'İ STREAM İLE OKU
# ============================================

print("="*60)
print("📥 INPUT.JSONL İŞLENİYOR")
print("="*60)

total_processed = 0
train_file = open("train.jsonl", "w", encoding="utf-8")

with open("input.jsonl", "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        item = json.loads(line.strip())
        
        print(f"\n[{line_num}] İşleniyor...")
        
        # URL veya text al
        if "url" in item:
            print(f"  📥 URL: {item['url']}")
            response = requests.get(item["url"], timeout=10)
            
            try:
                data = response.json()
                if "readme" in data:
                    content = f"{data.get('description', '')}\n\n{data.get('readme', '')[:2000]}"
                else:
                    content = json.dumps(data, ensure_ascii=False)[:2000]
            except:
                content = response.text[:2000]
        
        elif "text" in item:
            print(f"  📝 Text: {len(item['text'])} karakter")
            content = item["text"][:2000]
        
        else:
            print(f"  ⚠️ Geçersiz format, atlanıyor")
            continue
        
        # Teacher'a sor ve formatla
        formatted_data = process_content(content)
        
        # train.jsonl'e yaz (stream)
        for data in formatted_data:
            train_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        total_processed += len(formatted_data)
        print(f"  💾 {len(formatted_data)} örnek train.jsonl'e eklendi")

train_file.close()

print("\n" + "="*60)
print(f"✅ TAMAMLANDI! Toplam {total_processed} örnek")
print("="*60)
print("▶ Şimdi çalıştır: python index.py")
