Sen bir eğitim veri seti oluşturucususun. Aşağıdaki içerik hakkında {num_questions} adet soru-cevap çifti oluştur.

## İÇERİK:
{content}

## KURALLAR:
1. Sorular çeşitli olmalı (temel, kullanım, detay soruları)
2. Cevaplar SADECE verilen içeriğe dayanmalı
3. Kısa, net ve doğru cevaplar yaz
4. Türkçe yaz
5. JSON array formatında çıktı ver

## ÖRNEK:
```json
[
  {{"soru": "X nedir?", "cevap": "X, ... bir pakettir."}},
  {{"soru": "X nasıl kurulur?", "cevap": "npm install x komutu ile kurulur."}},
  {{"soru": "X ne işe yarar?", "cevap": "X, ... için kullanılır."}}
]
```

Şimdi {num_questions} adet soru-cevap üret (sadece JSON array):
