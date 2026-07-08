"""Gradio demo — Türkçe ürün yorumu duygu sınıflandırıcı.

Kaydedilmiş baseline modelini (TF-IDF + Lojistik Regresyon) yükler ve girilen
yorumu pozitif/negatif olarak sınıflandırıp güven skorlarını gösterir.

Yerel çalıştırma:
    pip install -r requirements.txt
    python app.py            # tarayıcıda http://127.0.0.1:7860

Hugging Face Spaces: bu dosya (app.py) + requirements.txt kök dizinde olduğu için
Space olarak doğrudan yayınlanabilir.

Not: Demo, hafif ve bağımlılıksız olması için baseline modelini kullanır.
BERTurk (macro-F1 0.86) modelini HF Hub'a yükledikten sonra bu demo onu kullanacak
şekilde kolayca güncellenebilir (aşağıdaki BERTURK_MODEL_ID'ye bakın).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from joblib import load

MODEL_PATH = Path(__file__).parent / "models" / "baseline_tfidf_logreg.joblib"
LABEL_NAMES = {0: "Negatif 👎", 1: "Pozitif 👍"}

# İleride BERTurk'ü Hub'a yükleyince buraya model kimliğini yazıp demoyu
# transformers pipeline'ına çevirebilirsiniz.
# BERTURK_MODEL_ID = "kullanici/berturk-turkish-sentiment"

if not MODEL_PATH.exists():
    raise SystemExit(
        f"Model bulunamadı: {MODEL_PATH}\n"
        "Önce baseline'ı eğitin: python -m src.baseline"
    )

_pipe = load(MODEL_PATH)


def classify(review: str):
    """Yorumu sınıflandır, her sınıf için güven skoru döndür (gr.Label formatı)."""
    review = (review or "").strip()
    if not review:
        return {"Metin girin": 1.0}
    proba = _pipe.predict_proba([review])[0]
    classes = _pipe.classes_
    return {LABEL_NAMES[int(c)]: float(p) for c, p in zip(classes, proba)}


EXAMPLES = [
    "Ürün çok kaliteli, kargo hızlıydı, kesinlikle tavsiye ederim.",
    "Paranıza yazık, iki gün sonra bozuldu. Almayın.",
    "Fena değil ama beklediğim gibi çıkmadı.",
    "Kötü diyemem, fiyatına göre gayet iyi.",
    "Rezalet bir ürün, kutusu bile ezikti.",
]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=4,
        label="Ürün yorumu",
        placeholder="Bir Türkçe ürün yorumu yazın...",
    ),
    outputs=gr.Label(num_top_classes=2, label="Tahmin"),
    examples=EXAMPLES,
    title="🇹🇷 Türkçe Ürün Yorumu Duygu Analizi",
    description=(
        "Bir ürün yorumunu **pozitif/negatif** sınıflandırır. "
        "Bu demo hafif **baseline** modelini (TF-IDF + Lojistik Regresyon) kullanır. "
        "Proje ayrıca BERTurk fine-tuning'i içerir (macro-F1 0.86). "
        "Kaynak: github.com/kucukenes17/turkce-urun-yorumu-duygu-analizi"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
