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

# Hata analizi: baseline'ın yanlış örnekleri (kategorize + örnekler)
python -m src.error_analysis
```
Çıktılar: `outputs/figures/*.png`, `outputs/eda_summary.md`,
`outputs/baseline_report.md`, `outputs/error_analysis.md` ve
`models/baseline_tfidf_logreg.joblib`.

BERTurk fine-tuning ise Colab'da yapılır (bkz. aşağıdaki not ve
`notebooks/berturk_finetuning.ipynb`).

## Sonuçlar
Her iki model de **aynı** stratified %80/%20 bölme (`random_state=42`) ve **aynı**
test kümesi (47.029 yorum) üzerinde değerlendirildi:

| Model | Accuracy | **Macro-F1** | F1 (negatif) | F1 (pozitif) |
|---|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 0.9086 | 0.7379 | 0.5264 | 0.9494 |
| **BERTurk (fine-tuned)** | **0.9688** | **0.8630** | **0.7425** | **0.9834** |

**BERTurk baseline'ı +0.1251 macro-F1 ile geçti.** En büyük kazanım azınlık
(negatif) sınıfında: negatif precision 0.39 → 0.78 (yanlış alarmlar yarı yarıya
azaldı), negatif F1 0.53 → 0.74.

**Neden accuracy tek başına yeterli değil?** Veri seti ~%94 pozitif; her şeye
"pozitif" diyen naif model %93.7 accuracy alır ama negatif sınıfta işe yaramaz.
Bu yüzden asıl ölçüt **macro-F1** ve confusion matrix.

## Hata Analizi (dürüst değerlendirme)
`python -m src.error_analysis` baseline'ın yanlışlarını inceler
(`outputs/error_analysis.md`). İki tür hata öne çıkıyor:

1. **Modelin gerçekten zorlandığı vakalar:** olumsuzlama ("kötü **değil**",
   "kötü diyemem"), karışık yorumlar ("güzel **ama** yapışkanı kötü"). Baseline
   kelime torbası olduğu için bağlamı kaçırır; BERTurk bunları büyük ölçüde çözer.
2. **Veri setinde etiket gürültüsü:** "gayet güzel, tavsiye ederim" gibi açıkça
   olumlu bazı yorumlar veride "negatif" etiketli — modelin değil, **etiketin**
   hatası. Yıldız-puanı → etiket eşlemesinin bir sınırlaması; tavan başarıyı bir
   miktar aşağı çeker.

## Proje Yapısı
```
src/
  config.py        # sabitler, yollar, etiket eşlemeleri
  data_loader.py   # HF'ten indir, önbelleğe al, DataFrame döndür
  explore.py       # EDA: dağılım, uzunluk, tekrar/null, örnekler, grafikler
  baseline.py      # TF-IDF + Lojistik Regresyon baseline
  error_analysis.py # baseline hatalarını kategorize eder + örnekler
notebooks/
  berturk_finetuning.ipynb  # Hafta 2: BERTurk fine-tuning (Colab GPU)
data/              # indirilen veri önbelleği (gitignore)
models/            # eğitilmiş modeller (gitignore)
outputs/           # üretilen grafik ve raporlar (gitignore)
```

## BERTurk Fine-tuning (Colab)
[`notebooks/berturk_finetuning.ipynb`](notebooks/berturk_finetuning.ipynb) —
Colab'da GPU ile `dbmdz/bert-base-turkish-cased` fine-tune eder. Notebook repoyu
klonlayıp **baseline ile birebir aynı** veriyi ve stratified bölmeyi kullanır,
sonunda macro-F1 üzerinden baseline ile otomatik karşılaştırma yapar.

**Çalıştırma:** Colab'da aç → Runtime → GPU → Run all. Hızlı deneme için
notebook'taki `MAX_TRAIN_SAMPLES` ayarını kullan.

## Sınırlamalar
- **Etiket gürültüsü:** etiketler yıldız puanından türetildiği için bir kısmı
  metinle çelişiyor (bkz. Hata Analizi).
- **Dengesiz veri:** ~%94 pozitif; azınlık sınıf performansı asıl darboğaz.
- **Kapsam:** yalnızca ikili (pozitif/negatif) sınıflandırma; nötr/aspect yok.
- BERTurk modeli boyutu nedeniyle repoya konmadı; notebook ile yeniden üretilir.

## Durum
- [x] **Hafta 1:** proje iskeleti + veri yükleme + keşif (EDA)
- [x] **Hafta 1:** TF-IDF + Lojistik Regresyon baseline
- [x] **Hafta 2:** BERTurk fine-tuning (Colab, macro-F1 = 0.863)
- [x] **Hafta 3:** değerlendirme + hata analizi + README

### Olası sonraki adımlar (stretch)
- Aspect-based sentiment ("neyin hakkında pozitif/negatif")
- Basit demo (Gradio) + BERTurk modelini HF Hub'a yükleme
- Etiket gürültüsünü temizleyip yeniden değerlendirme
