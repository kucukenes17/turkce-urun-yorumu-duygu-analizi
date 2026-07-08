<div align="center">

# ◆ Yorumetre

**Turkish product-review sentiment analysis** — a TF-IDF baseline vs. a fine-tuned
BERTurk transformer, with honest evaluation and a live demo that compares both models.

[🇹🇷 Türkçe README](README.md)

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-baseline-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-BERTurk-FFD21E)](https://huggingface.co/Eneskck/berturk-turkish-product-sentiment)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Macro-F1: `0.74` (baseline) → `0.86` (BERTurk)** &nbsp;·&nbsp; same test set

<img src="docs/images/demo-light.png" width="880" alt="Yorumetre demo UI">

</div>

---

## 🎯 Overview

An end-to-end NLP project comparing **two approaches** on ~235k real Turkish product
reviews, reported **honestly**:

| Model | Approach | Accuracy | **Macro-F1** | Negative F1 |
|:------|:---------|:--------:|:------------:|:-----------:|
| Baseline | TF-IDF + Logistic Regression | 0.9086 | 0.7379 | 0.5264 |
| **BERTurk** 🏆 | `dbmdz/bert-base-turkish-cased` fine-tune | **0.9688** | **0.8630** | **0.7425** |

> **BERTurk beats the baseline by +0.125 macro-F1.** The biggest gain is on the
> minority (negative) class, which is only ~6% of the data.

**Why macro-F1, not accuracy?** The dataset is ~94% positive. A naive "always positive"
model scores 93.7% accuracy yet is useless on the negative class. So the headline metric
is **macro-F1** plus the confusion matrix.

## ✨ Highlight: context matters

The same review, two answers — this is why the transformer wins:

> _"Kötü diyemem, fiyatına göre gayet iyi."_ ("Can't call it bad, quite good for the price" — actually **positive**)

| Model | Prediction | Correct? |
|:------|:-----------|:--------:|
| Baseline (bag of words) | Negative 99% | ❌ (trips on the word "kötü/bad") |
| **BERTurk** | Positive 77% | ✅ (understands the negation) |

## 🚀 Demo

```bash
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:7860
```

The demo shows **both models side by side** and highlights when they disagree. It loads
the model from the Hub ([`Eneskck/berturk-turkish-product-sentiment`](https://huggingface.co/Eneskck/berturk-turkish-product-sentiment))
and ships with light/dark themes. Being `app.py` + `requirements.txt` at the repo root,
it deploys to Hugging Face **Spaces** as-is.

## 📊 Dataset

[`fthbrmnby/turkish_product_reviews`](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews)
— ~235,165 real Turkish product reviews, binary labels (`0 = negative`, `1 = positive`).
**Heavily imbalanced (~94% positive)** — this drives every modeling choice (stratified
split, class weighting, macro-F1).

## 🔍 Honest error analysis

Two kinds of errors stand out:

1. **Genuinely hard cases** — negation ("not bad"), mixed reviews ("nice **but** fragile").
   The bag-of-words baseline misses these; **BERTurk largely solves them**.
2. **Label noise in the dataset** — some clearly positive reviews are labelled "negative".
   That's a label error, not a model error; it caps the achievable score. Star-rating →
   label mapping is the root cause.

## ⚙️ Setup & run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.data_loader        # download + cache data
python -m src.explore            # EDA + figures
python -m src.baseline           # train + evaluate baseline
python -m src.error_analysis     # inspect baseline errors
python -m src.evaluate_berturk   # evaluate BERTurk on the test set
python app.py                    # Gradio demo
```

BERTurk fine-tuning runs on Colab → [`notebooks/berturk_finetuning.ipynb`](notebooks/berturk_finetuning.ipynb).

## ⚠️ Limitations

- Label noise from star-rating-derived labels (see error analysis).
- Class imbalance (~94% positive); minority-class performance is the real bottleneck.
- Binary only (positive/negative); no neutral class or aspect-based sentiment.

## 📄 License

[MIT](LICENSE).
