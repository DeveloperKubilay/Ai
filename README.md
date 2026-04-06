# Otomatik Fine-Tuning Sistemi

Herhangi bir veri kaynağından soru-cevap üretip Qwen tabanlı küçük modelleri fine-tune eden düzenli proje yapısı.

## Hızlı Başlangıç

### 1. Veri gir
`data/input.jsonl` dosyasına URL veya text ekle:

```json
{"url": "https://registry.npmjs.org/elenora"}
{"text": "Uzun bir metin..."}
```

### 2. Soru-cevap üret
```bash
python scripts/create_data.py
```

### 3. Token dataset hazırla
```bash
python scripts/prepare_dataset.py
```

### 4. Eğit
```bash
python scripts/index.py
```

### 5. Test et
```bash
python scripts/quick_test.py
```

## Yapılandırma

Ana ayarlar `data/settings.json` içindedir.

## Klasör Yapısı

```text
data/
  input.jsonl
  settings.json
  train.jsonl
docs/
  amacımız.md
  teacher_prompt.md
  test.md
scripts/
  create_data.py
  prepare_dataset.py
  index.py
  quick_test.py
  pipeline_utils.py
qwen-trained-model/
checkpoints/
prepared-datasets/
```

## Notlar

- `prepared-datasets/` tokenize edilmiş ara veriyi tutar.
- `checkpoints/` eğitim sırasında resume için kullanılır.
- `qwen-trained-model/` final adapter çıktısıdır.

Daha detaylı açıklama için [docs/amacımız.md](./docs/amacımız.md) dosyasına bak.
