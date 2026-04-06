# Otomatik Fine-Tuning Sistemi

## Amaç

URL veya düz metinden soru-cevap verisi üretip Qwen tabanlı küçük modeli belirli bir konuda uzmanlaştırmak.

## Akış

### 1. Veri Girişi

Kaynak veri `data/input.jsonl` içindedir.

```json
{"url": "https://registry.npmjs.org/elenora"}
{"text": "Uzun bir metin..."}
```

### 2. Veri Üretimi

`python scripts/create_data.py`

- `data/input.jsonl` okunur
- içerik teacher modele gönderilir
- soru-cevaplar Qwen formatına çevrilir
- sonuç `data/train.jsonl` içine yazılır

### 3. Tokenization

`python scripts/prepare_dataset.py`

- `data/train.jsonl` tokenize edilir
- duplicate kayıtlar atılır
- çıktı `prepared-datasets/` altına yazılır
- yarıda kalırsa kaldığı yerden devam eder

### 4. Eğitim

`python scripts/index.py`

- hazır token dataset yüklenir
- otomatik eğitim profili hesaplanır
- checkpoint desteği ile eğitim başlar
- final model `qwen-trained-model/` altına kaydedilir

### 5. Test

`python scripts/quick_test.py`

## Dosya Yapısı

```text
data/
  input.jsonl
  settings.json
  train.jsonl
docs/
  amacımız.md
  test.md
  teacher_prompt.md
scripts/
  create_data.py
  prepare_dataset.py
  index.py
  quick_test.py
  pipeline_utils.py
```

## Notlar

- Ana yapılandırma `data/settings.json` içindedir.
- Teacher prompt `docs/teacher_prompt.md` içindedir.
- Checkpointler `checkpoints/` altında tutulur.
- Tokenized ara veri `prepared-datasets/` altında tutulur.
