Sen bir eğitim veri seti oluşturucususun. Aşağıdaki içerik hakkında {num_questions} adet eğitim örneği oluştur.

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
11. Her örnek `messages` alanı taşımalı
12. `messages` içinde roller sadece `user` ve `assistant` olsun; system mesajı ekleme
13. Gerekirse bir örnek birden fazla tur içerebilir ama son mesaj mutlaka `assistant` olmalı
14. Yorum, açıklama, markdown fence veya ekstra metin yazma

## ÖRNEK:
```json
[
  {{
    "messages": [
      {{"role": "user", "content": "X nedir?"}},
      {{"role": "assistant", "content": "X, ... bir pakettir."}}
    ]
  }},
  {{
    "messages": [
      {{"role": "user", "content": "X nasıl kurulur?"}},
      {{"role": "assistant", "content": "npm install x komutu ile kurulur."}}
    ]
  }}
]
```

Şimdi {num_questions} adet eğitim örneği üret (sadece JSON array):
