"""Gradio demo — Türkçe ürün yorumu duygu sınıflandırıcı.

Fine-tune edilmiş **BERTurk** modelini (Hugging Face Hub) kullanarak girilen
yorumu pozitif/negatif olarak sınıflandırır ve güven skorlarını gösterir.

Yerel çalıştırma:
    pip install -r requirements.txt
    python app.py            # tarayıcıda http://127.0.0.1:7860

Hugging Face Spaces: bu dosya (app.py) + requirements.txt kök dizinde olduğu için
Space olarak doğrudan yayınlanabilir.

Not: İlk çalıştırmada model (~400 MB) Hub'dan indirilir ve önbelleğe alınır.
"""

from __future__ import annotations

import gradio as gr
from transformers import pipeline

# Fine-tune edilmiş BERTurk modeli (Hafta 2, macro-F1 = 0.863).
MODEL_ID = "Eneskck/berturk-turkish-product-sentiment"

# Model etiketleri (id2label config'ten gelir) -> kullanıcıya gösterilen ad.
PRETTY = {
    "pozitif": "Pozitif 👍", "negatif": "Negatif 👎",
    "LABEL_1": "Pozitif 👍", "LABEL_0": "Negatif 👎",
}

_clf = pipeline("text-classification", model=MODEL_ID, top_k=None)


def classify(review: str):
    """Yorumu sınıflandır, her sınıf için güven skoru döndür (gr.Label formatı)."""
    review = (review or "").strip()
    if not review:
        return {"Metin girin": 1.0}
    scores = _clf(review)[0]  # top_k=None -> tüm sınıflar
    return {PRETTY.get(s["label"], s["label"]): float(s["score"]) for s in scores}


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
    title="🇹🇷 Türkçe Ürün Yorumu Duygu Analizi (BERTurk)",
    description=(
        "Bir ürün yorumunu **pozitif/negatif** sınıflandırır. "
        "Fine-tune edilmiş **BERTurk** modelini kullanır (macro-F1 0.86; "
        "baseline TF-IDF + LogReg 0.74 idi). "
        "Kaynak: github.com/kucukenes17/turkce-urun-yorumu-duygu-analizi"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
