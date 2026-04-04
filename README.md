# Otomatik Fine-Tuning Sistemi

Herhangi bir veri kaynağından (URL, text) otomatik olarak soru-cevap üretip, küçük dil modellerini (Qwen) fine-tune eden sistem.

## Özellikler

- 🤖 Teacher model ile otomatik soru-cevap üretimi
- 📦 Stream işleme (10GB+ veri destekler)
- ⚡ LoRA ile hızlı eğitim
- 🎯 Modüler ve temiz kod yapısı
- 🔧 Kolay yapılandırma (settings.json)

## Kurulum

```bash
pip install torch transformers datasets peft trl requests
```

## Hızlı Başlangıç

### 1. Ollama'yı Başlat
```bash
ollama run qwen2.5-coder:7b
```

### 2. Veri Hazırla
`input.jsonl` dosyasına URL veya text ekle:
```json
{"url": "https://registry.npmjs.org/elenora"}
{"text": "Uzun bir metin..."}
```

### 3. Soru-Cevap Üret
```bash
python create_data.py
```

### 4. Modeli Eğit
```bash
python index.py
```

### 5. Test Et
```bash
python quick_test.py
```

## Yapılandırma

`settings.json` dosyasını düzenle:

```json
{
  "model": {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "output_dir": "./qwen-trained-model"
  },
  "training": {
    "num_questions": 25,
    "num_epochs": 30,
    "learning_rate": 0.001
  },
  "teacher": {
    "model": "qwen2.5-coder:7b",
    "temperature": 0.8
  }
}
```

## Dosya Yapısı

```
├── input.jsonl           # Veri girişi
├── train.jsonl           # Üretilen eğitim verisi
├── settings.json         # Yapılandırma
├── teacher_prompt.md     # Teacher prompt
├── create_data.py        # Veri üretimi
├── index.py              # Model eğitimi
├── quick_test.py         # Test
└── qwen-trained-model/   # Eğitilmiş model
```

## Gereksinimler

- Python 3.8+
- CUDA destekli GPU (önerilir)
- Ollama (Teacher model için)
- ~8GB RAM (0.5B model için)

## Detaylı Dokümantasyon

Daha fazla bilgi için [amacımız.md](amacımız.md) dosyasına bakın.

## Lisans

MIT
