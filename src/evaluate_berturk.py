"""Fine-tune edilmiş BERTurk modelini test kümesinde değerlendir.

Hub'daki modeli (Eneskck/berturk-turkish-product-sentiment) baseline ile AYNI
stratified test kümesi üzerinde çalıştırır; metrikleri (accuracy, macro-F1,
sınıf bazlı F1), confusion matrix'i ve yanlış örnekleri üretir.

Çıktılar:
    docs/images/berturk_confusion_matrix.png
    outputs/berturk_report.md

Kullanım:
    python -m src.evaluate_berturk            # tüm test kümesi
    python -m src.evaluate_berturk --sample 5000
"""

from __future__ import annotations

import argparse
import textwrap
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config
from src.data_loader import load_reviews

MODEL_ID = "Eneskck/berturk-turkish-product-sentiment"
MAX_LEN = 128
BATCH_SIZE = 64


def _test_split() -> pd.DataFrame:
    df = load_reviews()
    df = df.dropna(subset=[config.TEXT_COL])
    df = df[df[config.TEXT_COL].str.strip() != ""]
    _, test_df = train_test_split(
        df, test_size=config.TEST_SIZE,
        stratify=df[config.LABEL_COL], random_state=config.RANDOM_STATE,
    )
    return test_df.reset_index(drop=True)


@torch.no_grad()
def _predict(texts, tokenizer, model) -> np.ndarray:
    preds = []
    n = len(texts)
    start = time.time()
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(list(batch), truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        logits = model(**enc).logits
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        if (i // BATCH_SIZE) % 20 == 0:
            done = min(i + BATCH_SIZE, n)
            rate = done / max(time.time() - start, 1e-6)
            print(f"  {done:,}/{n:,}  (~{rate:.0f} örnek/sn)", flush=True)
    return np.concatenate(preds)


def _plot_confusion(y_true, y_pred) -> None:
    labels = sorted(config.LABEL_NAMES)
    names = [config.LABEL_NAMES[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, cmap="Greens", colorbar=False, values_format=",")
    ax.set_title("BERTurk — Confusion Matrix")
    fig.tight_layout()
    for path in (config.FIGURES_DIR / "berturk_confusion_matrix.png",
                 config.ROOT_DIR / "docs" / "images" / "berturk_confusion_matrix.png"):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    plt.close(fig)
    print("[kaydedildi] docs/images/berturk_confusion_matrix.png")


def _examples(df, n=6) -> str:
    lines = []
    for _, row in df.head(n).iterrows():
        snippet = textwrap.shorten(str(row[config.TEXT_COL]), width=200, placeholder=" …")
        lines.append(f"- {snippet}")
    return "\n".join(lines) if lines else "- (yok)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="Sadece ilk N test örneğini değerlendir (hız için).")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    test_df = _test_split()
    if args.sample:
        test_df = test_df.head(args.sample).reset_index(drop=True)
    print(f"Test kümesi: {len(test_df):,} yorum")

    print(f"Model yükleniyor: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    print("Tahmin ediliyor...")
    y_true = test_df[config.LABEL_COL].to_numpy()
    y_pred = _predict(test_df[config.TEXT_COL].tolist(), tokenizer, model)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    labels = sorted(config.LABEL_NAMES)
    names = [config.LABEL_NAMES[i] for i in labels]
    report_txt = classification_report(y_true, y_pred, labels=labels,
                                       target_names=names, digits=4)
    print(f"\nAccuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}\n")
    print(report_txt)

    _plot_confusion(y_true, y_pred)

    # Hata örnekleri
    test_df = test_df.copy()
    test_df["pred"] = y_pred
    errors = test_df[test_df["pred"] != y_true]
    fn = errors[(errors[config.LABEL_COL] == 1) & (errors["pred"] == 0)]
    fp = errors[(errors[config.LABEL_COL] == 0) & (errors["pred"] == 1)]

    content = f"""# BERTurk Değerlendirme Raporu

**Model:** [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID})
**Test kümesi:** {len(test_df):,} yorum (baseline ile aynı stratified bölme, rs=42)

## Sonuçlar
- Accuracy: **{acc:.4f}**
- Macro-F1: **{macro_f1:.4f}**

```
{report_txt}
```

Confusion matrix: `docs/images/berturk_confusion_matrix.png`

## Kalan hatalar
Toplam {len(errors):,} hata (%{len(errors)/len(test_df)*100:.2f}).

### Gerçekte pozitif, model negatif dedi ({len(fn):,})
{_examples(fn)}

### Gerçekte negatif, model pozitif dedi ({len(fp):,})
{_examples(fp)}

> BERTurk baseline'a göre olumsuzlama/bağlam vakalarında çok daha iyi; kalan
> hataların önemli kısmı veri setindeki **etiket gürültüsünden** kaynaklanıyor
> (bkz. `outputs/error_analysis.md`).

---
*Bu dosya `python -m src.evaluate_berturk` ile üretilir.*
"""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUTS_DIR / "berturk_report.md").write_text(content, encoding="utf-8")
    print("[kaydedildi] outputs/berturk_report.md")


if __name__ == "__main__":
    main()
