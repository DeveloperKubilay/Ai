Sen bir eğitim verisi doğrulayıcısısın. Aşağıda kaynak içerik ve aday eğitim örnekleri veriliyor.

## KAYNAK:
{content}

## ADAY ÖRNEKLER:
{examples_json}

## GÖREV:
1. Sadece kaynak içeriğe açıkça dayanan örnekleri koru.
2. Yanlış, uydurma, tekrar eden, anlamsız veya yapay duran örnekleri sil.
3. Gerekirse cevabı düzelt ama yeni bilgi uydurma.
4. Her örnek `messages` alanı taşımalı.
5. `messages` içindeki roller sadece `user` ve `assistant` olsun.
6. Son mesaj mutlaka `assistant` olmalı.
7. Yalnızca temizlenmiş JSON array döndür.
8. Açıklama, markdown fence veya ekstra metin yazma.

## ÖRNEK ÇIKTI:
```json
[
  {
    "messages": [
      {"role": "user", "content": "X nedir?"},
      {"role": "assistant", "content": "X, ... bir pakettir."}
    ]
  }
]
```

Şimdi sadece temizlenmiş JSON array döndür:
