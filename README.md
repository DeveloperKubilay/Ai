# Otomatik Fine-Tuning Sistemi

Bu repo mesaj tabanlı veri üretimi, token hazırlama, LoRA fine-tuning, hızlı test ve red-team repair akışını tek yerde toplar.

## Kısa Mantık

Akış sırası şudur:

1. `data/input.jsonl` içinden kaynaklar okunur.
2. `scripts/create_data.py` eğitim verisini `data/train.jsonl` içine yazar.
3. `scripts/prepare_dataset.py` bunu tokenize edip `prepared-datasets/` altına hazırlar.
4. `scripts/index.py` modeli eğitir ve `checkpoints/` altında ara kayıtlar tutar.
5. `scripts/quick_test.py` hızlı smoke test yapar.
6. `scripts/redteam_repair.py` modeli zorlayıp hata bulursa `data/repair_train.jsonl` üretir.

## Hızlı Başlangıç

### 1. Kaynak veriyi tanımla
`data/input.jsonl` dosyasına satır satır kaynak eklenir:

```json
{"file": "data/data.jsonl", "ai": false}
{"text": "Uzun bir metin..."}
{"url": "https://ornek-site.com/dokuman"}
```

### 2. Eğitim verisini üret

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

### 5. Hızlı test

```bash
python scripts/quick_test.py
```

### 6. Red-team ve repair

```bash
python scripts/redteam_repair.py --apply
```

### 7. Tek komutluk akış

```bash
python scripts/full_pipeline.py
```

## Önemli Not

Şu anki ayarlar ve seed veri sistemi genel bir AI gibi değil, daha çok alan bilgisine dayanan bir bot gibi davranmaya iter.

Bunun ana sebebi:

- `data/settings.json` içindeki `system_prompt`
- `data/data.jsonl` içindeki örneklerin tek alan etrafında dönmesi

Yani bugün repo olduğu haliyle "genel AI + her konuda doğal cevap" değil, "verilen bilgi alanına sadık cevap" davranışına daha yakındır.

## Temel Dosyalar

### data/

- `data/input.jsonl`: Ana giriş listesidir. Hangi kaynaklardan veri üretileceğini söyler.
- `data/data.jsonl`: Hazır eğitim örnekleri içeren ana seed veri dosyasıdır. `ai:false` ise buradaki kayıtlar doğrudan kullanılır.
- `data/train.jsonl`: Eğitime girecek derlenmiş son veri dosyasıdır.
- `data/settings.json`: Model, eğitim, inference, teacher, verification ve checkpoint ayarlarını tutar.
- `data/redteam_report.jsonl`: Red-team testinden çıkan rapordur. Model hangi sorularda bozulmuş gösterir.
- `data/repair_train.jsonl`: Red-team sonrası üretilen düzeltme setidir. Yani eğitimden sonra sonradan bulunan tamir örnekleri burada tutulur.

### docs/

- `docs/teacher_prompt.md`: Ham metinden veya URL içeriğinden eğitim örneği üretmesi için teacher modele verilen prompttur.
- `docs/teacher_verify_prompt.md`: Teacher'ın ürettiği örnekleri istersek ikinci kez temizlemek için kullanılan isteğe bağlı prompttur.
- `docs/redteam_prompt.md`: Eğitilmiş modeli zorlamak için test soruları üreten prompttur.
- `docs/redteam_verify_prompt.md`: Red-team testinde modelin verdiği cevapların doğru mu bozuk mu olduğuna karar veren prompttur.
- `docs/test.md`: Geçici notlar veya fikir karalamaları için kullanılan dosyadır.

### scripts/

- `scripts/create_data.py`: `input.jsonl` içindeki kaynaklardan `train.jsonl` üretir.
- `scripts/prepare_dataset.py`: `train.jsonl` verisini tokenize eder ve hazır dataset yazar.
- `scripts/index.py`: Eğitimi başlatır, checkpoint yönetir ve final adapter'ı kaydeder.
- `scripts/quick_test.py`: Soruları tek tek ve ayrı ayrı soran hızlı test scriptidir. Mesaj geçmişi taşımaz.
- `scripts/redteam_repair.py`: Modeli zorlar, rapor çıkarır, gerekiyorsa repair set üretir.
- `scripts/full_pipeline.py`: Veri üretiminden eğitime kadar ana adımları tek komutta çalıştırır.
- `scripts/util/pipeline_utils.py`: Ortak yardımcı fonksiyonların toplandığı yerdir. Path çözümü, mesaj normalize etme, runtime tespiti ve profil hesapları burada durur.
- `scripts/teacher_client.py`: Teacher veya verifier modele HTTP isteği atan küçük istemcidir. JSON array parse etme ve retry mantığı burada bulunur.
- `scripts/util/model_runtime.py`: Yerel fine-tuned modeli yüklemek ve cevap üretmek için ortak inference katmanıdır. `quick_test.py` ve `redteam_repair.py` aynı yükleme mantığını tekrar yazmasın diye ayrıdır.

## Sık Karışan Dosyalar

### `teacher_verify_prompt.md` ne işe yarar?

Bu dosya eğitimden önce devreye girebilir.

Amaç:

- teacher'ın ürettiği örnekleri kontrol etmek
- bozuk JSON veya uydurma cevapları elemek
- sadece temiz örnekleri `train.jsonl` içine sokmak

Kısaca: ikinci güvenlik katmanı.

Not:

- Teacher prompt iyi olsa bile model bazen format bozabilir veya kaynak dışına çıkabilir.
- Bu yüzden verify adımı vardır.
- Şu an karışıklığı azaltmak için varsayılan olarak kapalıdır. İstersen `data/settings.json` içinden tekrar açılabilir.

### `teacher_client.py` neden var?

Teacher veya verifier modele istek atma işi tek yerde toplansın diye var.

İçinde şunlar bulunur:

- endpoint'e istek atma
- cevap içinden JSON array çıkarma
- cevap bozuksa retry etme

Kısaca: teacher ile konuşan küçük yardımcı katman.

### `model_runtime.py` neden ayrı dosya?

Model yükleme ve cevap üretme kodu hem testte hem red-team tarafında lazımdı.
Her dosyada aynı kodu kopyalamamak için ayrıldı.

Kısaca:

- modeli yükler
- tokenizer yükler
- verilen mesajları modele sorar
- cevabı döndürür

### `repair_train.jsonl` ne işe yarar?

`repair_train.jsonl`:

- model eğitildikten sonra bulunan hatalardan üretilir
- sonradan eklenen tamir setidir
- red-team sonucu oluşur

Kısaca: eğitim sonrası bulunan hataları düzeltmek için sonradan biriken ek veri.

## Diğer Klasörler

- `prepared-datasets/`: tokenize edilmiş hazır datasetler
- `checkpoints/`: eğitim sırasında ara kayıtlar
- `qwen-trained-model/`: final adapter çıktısı

## Kullandığın Teknoloji Hakkında

Bu repo Hugging Face kütüphanelerini yerelde çalıştırır. Veri dosyalarının dışarı otomatik yüklenmesi için özel bir mekanizma yoktur.

Yine de uzak model ID kullanırsan ilk model indirme sırasında internet erişimi gerekir. Tam kapalı kullanım istersen model ve tokenizer'ı yerel path'ten çalıştırman gerekir.
