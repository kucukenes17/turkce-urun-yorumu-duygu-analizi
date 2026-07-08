# Türkçe Ürün Yorumu Duygu Sınıflandırıcı

Türkçe ürün yorumlarını **pozitif / negatif** olarak sınıflandıran bir NLP projesi.
Hedef: baseline (TF-IDF + Lojistik Regresyon) ile transformer (BERTurk) karşılaştırması
ve dürüst hata analizi. Proje planı için bkz. [PROJECT.md](PROJECT.md).

## Veri Seti
[`fthbrmnby/turkish_product_reviews`](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews)
— ~235k Türkçe ürün yorumu, ikili etiket (`0 = negatif`, `1 = pozitif`).
**Not:** veri seti ciddi dengesizdir (~%94 pozitif); ölçüt olarak accuracy yerine
F1 + confusion matrix kullanılır.

## Kurulum
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```
(Linux/macOS: `source .venv/bin/activate`)

## Çalıştırma
```powershell
# Veriyi indir + yerel parquet önbelleğine yaz
python -m src.data_loader

# Keşifçi veri analizi (EDA): konsol özeti + outputs/ altına grafik ve rapor
python -m src.explore

# Baseline model: TF-IDF + Lojistik Regresyon (stratified split, class_weight)
python -m src.baseline
```
Çıktılar: `outputs/figures/*.png`, `outputs/eda_summary.md`,
`outputs/baseline_report.md` ve `models/baseline_tfidf_logreg.joblib`.

## Baseline Sonuçları
TF-IDF (1-2 gram) + LogisticRegression(`class_weight="balanced"`), stratified %80/%20 bölme:

| Ölçüt | Değer |
|---|---|
| Accuracy | 0.9086 |
| **Macro-F1** | **0.7379** |
| F1 (pozitif) | 0.9494 |
| F1 (negatif) | 0.5264 |

**Önemli not (dürüst kıyas):** accuracy (0.909), her şeye "pozitif" diyen naif
modelin accuracy'sinden (0.937) *düşük*. Bu bir hata değil — `class_weight="balanced"`
azınlık (negatif) sınıfını yakalamak için biraz pozitif doğruluğundan feda ediyor.
Bu yüzden asıl ölçüt **macro-F1**. BERTurk'ün bu 0.738'i geçmesi hedeflenecek.
Ayrıntı: `outputs/baseline_report.md`.

## Proje Yapısı
```
src/
  config.py        # sabitler, yollar, etiket eşlemeleri
  data_loader.py   # HF'ten indir, önbelleğe al, DataFrame döndür
  explore.py       # EDA: dağılım, uzunluk, tekrar/null, örnekler, grafikler
  baseline.py      # TF-IDF + Lojistik Regresyon baseline
data/              # indirilen veri önbelleği (gitignore)
models/            # eğitilmiş modeller (gitignore)
outputs/           # üretilen grafik ve raporlar (gitignore)
```

## Durum
- [x] **Hafta 1:** proje iskeleti + veri yükleme + keşif (EDA)
- [x] **Hafta 1:** TF-IDF + Lojistik Regresyon baseline
- [ ] Hafta 2: BERTurk fine-tuning (Colab GPU)
- [ ] Hafta 3: değerlendirme + hata analizi + demo
