"""Proje geneli sabitler ve yollar.

Tüm yollar repo köküne göre çözülür; script hangi dizinden çalıştırılırsa
çalıştırılsın aynı konumları gösterir.
"""

from pathlib import Path

# --- Veri seti ---
# Hugging Face Hub kimliği: ~235k Türkçe ürün yorumu, ikili etiket.
DATASET_ID = "fthbrmnby/turkish_product_reviews"

# Ham veri setindeki orijinal sütun adları.
SRC_TEXT_COL = "sentence"
SRC_LABEL_COL = "sentiment"

# Proje içinde kullanılan normalize edilmiş sütun adları.
TEXT_COL = "text"
LABEL_COL = "label"
LABEL_NAME_COL = "label_name"

# 0 = negatif, 1 = pozitif (veri setinin kendi kodlaması).
LABEL_NAMES = {0: "negatif", 1: "pozitif"}

# --- Yollar ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# İndirilen ham verinin yerel parquet önbelleği (tekrar indirmeyi önler).
RAW_CACHE = DATA_DIR / "turkish_product_reviews.parquet"

# EDA metinsel özet raporu.
EDA_SUMMARY = OUTPUTS_DIR / "eda_summary.md"

MODELS_DIR = ROOT_DIR / "models"

# --- Baseline (TF-IDF + Lojistik Regresyon) ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
# TF-IDF: 1-2 gram, çok nadir ve çok yaygın terimleri ele.
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.9
TFIDF_MAX_FEATURES = 50_000

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"
BASELINE_REPORT = OUTPUTS_DIR / "baseline_report.md"
