"""Baseline model: TF-IDF + Lojistik Regresyon.

Transformer'a (BERTurk) geçmeden önceki kıyas noktası. Veri seti dengesiz
olduğu için:
  - eğitim/test bölmesi stratified,
  - LogisticRegression class_weight="balanced",
  - ölçüt olarak accuracy DEĞİL, macro-F1 + sınıf bazlı F1 + confusion matrix.

Konsola metrikleri basar, modeli models/ altına, raporu ve confusion matrix
grafiğini outputs/ altına kaydeder.

Kullanım:
    python -m src.baseline
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import load_reviews


def _prepare_data() -> pd.DataFrame:
    """Veriyi yükle, boş/eksik metinleri at."""
    df = load_reviews()
    before = len(df)
    df = df.dropna(subset=[config.TEXT_COL])
    df = df[df[config.TEXT_COL].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"[bilgi] {dropped} boş/eksik metin çıkarıldı.")
    return df


def _build_pipeline() -> Pipeline:
    """TF-IDF + Lojistik Regresyon pipeline'ı."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            max_features=config.TFIDF_MAX_FEATURES,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # dengesizliği telafi et
            random_state=config.RANDOM_STATE,
        )),
    ])


def _plot_confusion(y_true, y_pred) -> None:
    labels = sorted(config.LABEL_NAMES)  # [0, 1]
    names = [config.LABEL_NAMES[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=","
    )
    ax.set_title("Baseline — Confusion Matrix")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "baseline_confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[kaydedildi] {path}")


def _write_report(acc, macro_f1, report_txt, majority_acc) -> None:
    content = f"""# Baseline Raporu — TF-IDF + Lojistik Regresyon

**Veri seti:** `{config.DATASET_ID}`
**Bölme:** stratified, test oranı = {config.TEST_SIZE}, random_state = {config.RANDOM_STATE}
**Model:** TF-IDF (ngram={config.TFIDF_NGRAM_RANGE}, min_df={config.TFIDF_MIN_DF},
max_features={config.TFIDF_MAX_FEATURES}) + LogisticRegression(class_weight="balanced")

## Sonuçlar (test kümesi)
- Accuracy: **{acc:.4f}**
- Macro-F1: **{macro_f1:.4f}**

> Kıyas: her şeye çoğunluk sınıfı ("pozitif") diyen naif model accuracy ≈
> **{majority_acc:.4f}** alır ama azınlık (negatif) sınıfında F1 = 0 olur.
> Bu yüzden **macro-F1** asıl ölçüttür.

### Sınıf bazlı rapor
```
{report_txt}
```

Confusion matrix: `outputs/figures/baseline_confusion_matrix.png`

---
*Bu dosya `python -m src.baseline` ile otomatik üretilir.*
"""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.BASELINE_REPORT.write_text(content, encoding="utf-8")
    print(f"[kaydedildi] {config.BASELINE_REPORT}")


def main() -> None:
    df = _prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df[config.TEXT_COL],
        df[config.LABEL_COL],
        test_size=config.TEST_SIZE,
        stratify=df[config.LABEL_COL],
        random_state=config.RANDOM_STATE,
    )
    print(f"Eğitim: {len(X_train):,} | Test: {len(X_test):,}")

    pipe = _build_pipeline()
    print("Model eğitiliyor (TF-IDF + LogisticRegression)...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    labels = sorted(config.LABEL_NAMES)
    names = [config.LABEL_NAMES[i] for i in labels]
    report_txt = classification_report(
        y_test, y_pred, labels=labels, target_names=names, digits=4
    )
    # Naif çoğunluk-sınıfı kıyası.
    majority_acc = y_test.value_counts(normalize=True).max()

    print("\n=== Baseline sonuçları (test) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"(Naif çoğunluk-sınıfı accuracy ≈ {majority_acc:.4f})")
    print("\n" + report_txt)

    _plot_confusion(y_test, y_pred)
    _write_report(acc, macro_f1, report_txt, majority_acc)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipe, config.BASELINE_MODEL_PATH)
    print(f"[kaydedildi] {config.BASELINE_MODEL_PATH}")

    print("\nBaseline tamamlandı.")


if __name__ == "__main__":
    main()
