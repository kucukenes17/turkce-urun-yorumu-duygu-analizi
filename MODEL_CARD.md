---
language: tr
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: dbmdz/bert-base-turkish-cased
datasets:
  - fthbrmnby/turkish_product_reviews
tags:
  - sentiment-analysis
  - turkish
  - text-classification
  - bert
metrics:
  - f1
  - accuracy
widget:
  - text: "Ürün çok kaliteli, kargo hızlıydı, kesinlikle tavsiye ederim."
  - text: "Kötü diyemem, fiyatına göre gayet iyi."
  - text: "Paranıza yazık, iki gün sonra bozuldu."
---

# BERTurk — Türkçe Ürün Yorumu Duygu Sınıflandırıcı

`dbmdz/bert-base-turkish-cased` modelinin, Türkçe ürün yorumlarını **pozitif / negatif**
sınıflandırmak için fine-tune edilmiş hâli. [**Yorumetre**](https://github.com/kucukenes17/turkce-urun-yorumu-duygu-analizi)
projesinin ana modelidir.

- **Etiketler:** `0 = negatif`, `1 = pozitif`
- **Temel model:** dbmdz/bert-base-turkish-cased
- **Veri:** [fthbrmnby/turkish_product_reviews](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews) (~235k yorum)
- **Dil:** Türkçe

## Sonuçlar (test kümesi, 47.029 yorum)

| Ölçüt | Değer |
|---|---|
| Accuracy | 0.9688 |
| **Macro-F1** | **0.8630** |
| F1 (negatif) | 0.7425 |
| F1 (pozitif) | 0.9834 |

Kıyas için TF-IDF + Lojistik Regresyon baseline aynı test kümesinde macro-F1 **0.7379**
alır; bu model onu **+0.125** geçer. En büyük kazanım, verinin ~%6'sını oluşturan
azınlık (negatif) sınıfındadır.

> Veri seti ~%94 pozitif olduğundan accuracy yanıltıcıdır; asıl ölçüt **macro-F1**'dir.

## Kullanım

```python
from transformers import pipeline

clf = pipeline("text-classification", model="Eneskck/berturk-turkish-product-sentiment")
print(clf("Kötü diyemem, fiyatına göre gayet iyi."))
# [{'label': 'pozitif', 'score': 0.99...}]
```

## Eğitim

- Stratified %80/%20 bölme (`random_state=42`), baseline ile birebir aynı split.
- Weighted cross-entropy (sınıf dengesizliği telafisi).
- 2 epoch, learning rate 2e-5, max_length 128, batch 32, fp16 (Colab T4 GPU).

## Sınırlamalar

- **Etiket gürültüsü:** etiketler yıldız puanından türetildiği için bir kısmı metinle
  çelişir (bazı açıkça olumlu yorumlar "negatif" etiketli). Bu, raporlanan tavan başarıyı
  bir miktar aşağı çeker.
- **Dengesiz veri:** azınlık (negatif) sınıf performansı asıl darboğazdır.
- **Kapsam:** yalnızca ikili (pozitif/negatif); nötr sınıf veya aspect-based analiz yoktur.
- Yalnızca ürün yorumu alanında değerlendirilmiştir; farklı alanlarda (haber, sosyal medya)
  performans düşebilir.

## Atıf

Proje ve kaynak kod: https://github.com/kucukenes17/turkce-urun-yorumu-duygu-analizi
