# BERTurk Değerlendirme Raporu

**Model:** [`Eneskck/berturk-turkish-product-sentiment`](https://huggingface.co/Eneskck/berturk-turkish-product-sentiment)
**Test kümesi:** 47.029 yorum (baseline ile aynı stratified bölme, `random_state=42`)
**Değerlendirme:** tam test kümesi, Colab GPU (fine-tuning ile aynı ortam).

## Sonuçlar (tam test kümesi)
- Accuracy: **0.9688**
- Macro-F1: **0.8630**

```
              precision    recall  f1-score   support

     negatif     0.7785    0.7097    0.7425      2976
     pozitif     0.9805    0.9864    0.9834     44053

    accuracy                         0.9688     47029
   macro avg     0.8795    0.8480    0.8630     47029
weighted avg     0.9677    0.9688    0.9682     47029
```

Confusion matrix (`docs/images/berturk_confusion_matrix.png`):

|              | pred negatif | pred pozitif |
|:-------------|:------------:|:------------:|
| **true negatif** | 2.112 | 864 |
| **true pozitif** | 599 | 43.454 |

Baseline ile kıyas: negatif F1 **0.53 → 0.74**, negatif precision **0.39 → 0.78**.

## Kalan hataların doğası
BERTurk, baseline'ın kaçırdığı olumsuzlama ve bağlam vakalarını büyük ölçüde
çözüyor. Kalan hataların önemli bir kısmı ise **modelin değil, etiketin hatası** —
yani veri setindeki gürültü. Test kümesinden gerçek örnekler:

**Gerçekte "negatif" etiketli ama model (haklı olarak) pozitif dedi:**
- _"piyasa fiyatına ayağınıza gelsin avantajlı"_ → açıkça olumlu, yanlış etiket
- _"ürün 2 gün içinde elime ulaştı teşekkürler ... çanta fiyatına göre gayet güzel"_ → olumlu
- _"ilkokula yeni başlayan yeğenime aldığım mikroskop gayet güzel çıktı ... tavsiye ederim"_ → olumlu

**Gerçekten zor (bağlam/karışık) vakalar:**
- _"ürün güzel ancak demliğindeki süzgeç tamamen düşünülmeden yapılmış ..."_ → karışık yorum
- _"silikon kılıf tam oturdu ve şık duruyor fakat kırılmaz cam ekranı tam kaplamıyor ..."_ → olumlu + şikayet

> **Sonuç:** raporlanan macro-F1 (0.863) muhtemelen modelin gerçek tavanının biraz
> altında; çünkü mükemmel bir model bile yanlış etiketli örnekleri "yanlış" bilmek
> zorunda. Bkz. baseline hata analizi: `outputs/error_analysis.md`.

---
*Metrikler tam test kümesinden; `python -m src.evaluate_berturk` ile yeniden üretilebilir
(CPU'da yavaş; `--sample N` ile hızlı bir alt-küme değerlendirmesi yapılabilir).*
