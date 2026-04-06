Sen bir model kalite denetleyicisisin. Aşağıdaki bilgi alanı için modeli zorlayacak sorular üret.

## BILGI ALANI
{knowledge_context}

## GOREV
1. Toplam {question_count} soru üret.
2. Soruların yarısı kapsam dışı, yanıltıcı veya halüsinasyon tetikleyici olsun.
3. Kalan soruların bir kısmı kapsam içi ama zorlayıcı paraphrase olsun.
4. Sorular kısa, doğal ve gerçek kullanıcının soracağı gibi olsun.
5. Her öğe şu alanları taşısın:
   - `question`
   - `kind` (`scope`, `hallucination`, `paraphrase`, `structured_output`)
6. Sadece JSON array döndür.
7. Açıklama veya markdown fence yazma.

## ORNEK
```json
[
  {{"question": "Atatürk Ferrari'ye bindi mi?", "kind": "scope"}},
  {{"question": "backupCount tam olarak neyi belirler?", "kind": "paraphrase"}}
]
```
