# Baseline Raporu — TF-IDF + Lojistik Regresyon

**Veri seti:** `fthbrmnby/turkish_product_reviews`
**Bölme:** stratified, test oranı = 0.2, random_state = 42
**Model:** TF-IDF (ngram=(1, 2), min_df=5,
max_features=50000) + LogisticRegression(class_weight="balanced")

## Sonuçlar (test kümesi)
- Accuracy: **0.9086**
- Macro-F1: **0.7379**

> Kıyas: her şeye çoğunluk sınıfı ("pozitif") diyen naif model accuracy ≈
> **0.9367** alır ama azınlık (negatif) sınıfında F1 = 0 olur.
> Bu yüzden **macro-F1** asıl ölçüttür.

### Sınıf bazlı rapor
```
              precision    recall  f1-score   support

     negatif     0.3917    0.8024    0.5264      2976
     pozitif     0.9856    0.9158    0.9494     44053

    accuracy                         0.9086     47029
   macro avg     0.6887    0.8591    0.7379     47029
weighted avg     0.9480    0.9086    0.9227     47029

```

Confusion matrix: `outputs/figures/baseline_confusion_matrix.png`

---
*Bu dosya `python -m src.baseline` ile otomatik üretilir.*
