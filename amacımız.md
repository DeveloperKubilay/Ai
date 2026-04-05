# Otomatik Fine-Tuning Sistemi

## Amaç
Herhangi bir veri kaynağından (URL, text) otomatik olarak soru-cevap üretip, küçük dil modellerini (Qwen) fine-tune etmek.

## Sistem Mimarisi

### 1. Veri Girişi (input.jsonl)
Kullanıcı buraya URL veya text girer:
```json
{"url": "https://registry.npmjs.org/elenora"}
{"text": "Uzun bir metin..."}
{"url": "https://example.com/api/data"}
```

- 10GB bile olsa sorun yok (stream ile işlenir)
- URL ve text karışık olabilir
- Her satır bağımsız işlenir

### 2. Veri İşleme (create_data.py)
**Akış:**
1. input.jsonl'i stream ile oku (RAM dolmaz)
2. Her satır için:
   - URL ise → İçeriği çek
   - Text ise → Direkt kullan
3. İçeriği Teacher modele gönder (Ollama)
4. Teacher, N adet soru-cevap üretir
5. Qwen chat formatına çevir:
   ```
   <|im_start|>user
   Soru burada<|im_end|>
   <|im_start|>assistant
   Cevap burada<|im_end|>
   ```
6. train.jsonl'e yaz (stream)

**Özellikler:**
- 3 kez deneme (hata durumunda)
- JSON parse hataları yakalanır
- Boş soru-cevaplar atlanır
- Her satır işlenirken progress gösterir

### 3. Model Eğitimi (index.py)
**Akış:**
1. train.jsonl'i yükle
2. Qwen modelini yükle (0.5B veya 1.5B)
3. LoRA uygula (efficient fine-tuning)
4. Eğit
5. Modeli kaydet

**Özellikler:**
- LoRA ile hızlı eğitim
- FP16 precision (GPU optimizasyonu)
- Gradient accumulation
- Warmup + AdamW optimizer

### 4. Test (quick_test.py)
Eğitilmiş modeli test et:
```python
python quick_test.py
```

## Yapılandırma (settings.json)

```json
{
  "url": "https://...",           // Varsayılan URL (input.jsonl override eder)
  "model": {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",  // 0.5B veya 1.5B
    "output_dir": "./qwen-trained-model"
  },
  "training": {
    "num_questions": 25,          // Her içerik için kaç soru
    "num_epochs": 30,             // Eğitim epoch sayısı
    "learning_rate": 0.001,       // Öğrenme hızı
    "batch_size": 2,
    "lora_r": 16,                 // LoRA rank
    "lora_alpha": 32
  },
  "teacher": {
    "url": "http://localhost:11434/api/generate",
    "model": "qwen2.5-coder:7b",  // Ollama model
    "temperature": 0.8            // Yaratıcılık (0.7-0.9)
  }
}
```

## Teacher Prompt (teacher_prompt.md)
Teacher modelin nasıl soru-cevap üreteceğini belirler.

**Özelleştirme:**
- Soru tiplerini değiştir
- Cevap formatını ayarla
- Dil değiştir (Türkçe/İngilizce)

## Kullanım

### Adım 1: Veri Hazırla
```bash
# input.jsonl'e URL veya text ekle
echo '{"url": "https://registry.npmjs.org/elenora"}' > input.jsonl
```

### Adım 2: Soru-Cevap Üret
```bash
python create_data.py
# → train.jsonl oluşturulur
```

### Adım 3: Eğit
```bash
python index.py
# → qwen-trained-model/ klasörü oluşturulur
```

### Adım 4: Test
```bash
python quick_test.py
```

## Avantajlar

1. **Ölçeklenebilir**: 10GB veri bile işlenebilir (stream)
2. **Modüler**: Her dosya tek bir iş yapıyor
3. **Esnek**: URL, text, farklı formatlar desteklenir
4. **Otomatik**: Teacher model soru-cevap üretiyor
5. **Hızlı**: LoRA ile efficient training
6. **Özelleştirilebilir**: Prompt, model, parametreler ayarlanabilir

## Dosya Yapısı

```
├── input.jsonl              # Veri girişi (URL/text)
├── train.jsonl              # Üretilen eğitim verisi
├── settings.json            # Yapılandırma
├── teacher_prompt.md        # Teacher prompt
├── create_data.py           # Veri üretimi
├── index.py                 # Model eğitimi
├── quick_test.py            # Test
└── qwen-trained-model/      # Eğitilmiş model
```

## Notlar

- **Teacher Model**: Ollama'da çalışmalı (`ollama run qwen2.5-coder:7b`)
- **GPU**: CUDA destekli GPU önerilir (CPU'da çok yavaş)
- **RAM**: 0.5B model için ~4GB, 1.5B için ~8GB
- **Disk**: Model + veri için ~5-10GB

## DeepSeek Yaklaşımı

Sistem, DeepSeek'in kullandığı self-improvement yaklaşımını kullanır:
1. Güçlü bir model (Teacher) zayıf modeli (Student) eğitir
2. Teacher, veriyi analiz edip soru-cevap üretir
3. Student bu verilerle fine-tune edilir
4. Sonuç: Küçük model, spesifik konuda uzmanlaşır

## Gelecek İyileştirmeler

- [ ] Verification step (Teacher cevapları doğrular)
- [ ] Farklı model formatları (Llama, Mistral)
- [ ] Otomatik hyperparameter tuning
