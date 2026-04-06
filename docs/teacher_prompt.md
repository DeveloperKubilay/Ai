Sen bir eğitim veri seti oluşturucususun. Aşağıdaki içerik hakkında {num_questions} adet soru-cevap çifti oluştur.

## İÇERİK:
{content}

## KURALLAR:
1. Sorular çeşitli olmalı (temel, kullanım, detay soruları)
2. Cevaplar SADECE verilen içeriğe dayanmalı
3. Kısa, net ve doğru cevaplar yaz
4. Türkçe yaz
5. JSON array formatında çıktı ver
6. Aynı soruyu veya aynı anlamdaki soruyu tekrar etme
7. Yapay, bozuk veya anlamsız soru yazma
8. Sayısal değer, varsayılan ayar veya API imzası geçiyorsa aynen koru
9. Sorular gerçek kullanıcının soracağı kadar doğal olsun
10. Cevaplar tek cümlelik ve öğretici olsun; gereksiz laf uzatma

## ÖRNEK:
```json
[
  {{"soru": "X nedir?", "cevap": "X, ... bir pakettir."}},
  {{"soru": "X nasıl kurulur?", "cevap": "npm install x komutu ile kurulur."}},
  {{"soru": "X ne işe yarar?", "cevap": "X, ... için kullanılır."}}
]
```

Şimdi {num_questions} adet soru-cevap üret (sadece JSON array):
