"""Gradio demo — Türkçe ürün yorumu duygu sınıflandırıcı.

İki modeli **yan yana** karşılaştırır:
  - BERTurk (fine-tuned, macro-F1 0.86)  ← ana model
  - Baseline (TF-IDF + Lojistik Regresyon, macro-F1 0.74)  ← kıyas

Böylece transformer'ın baseline'a üstünlüğü canlı olarak görülebilir
(özellikle olumsuzlama / bağlam içeren zor yorumlarda).

Yerel çalıştırma:
    pip install -r requirements.txt
    python app.py            # tarayıcıda http://127.0.0.1:7860

Hugging Face Spaces: app.py + requirements.txt kök dizinde olduğu için doğrudan
Space olarak yayınlanabilir.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from joblib import load
from transformers import pipeline

BERTURK_MODEL_ID = "Eneskck/berturk-turkish-product-sentiment"
BASELINE_PATH = Path(__file__).parent / "models" / "baseline_tfidf_logreg.joblib"
GITHUB_URL = "https://github.com/kucukenes17/turkce-urun-yorumu-duygu-analizi"

PRETTY = {
    "pozitif": "Pozitif 👍", "negatif": "Negatif 👎",
    "LABEL_1": "Pozitif 👍", "LABEL_0": "Negatif 👎",
    1: "Pozitif 👍", 0: "Negatif 👎",
}

# --- Modelleri yükle ---
_berturk = pipeline("text-classification", model=BERTURK_MODEL_ID, top_k=None)
_baseline = load(BASELINE_PATH) if BASELINE_PATH.exists() else None


def _berturk_scores(text: str) -> dict:
    scores = _berturk(text)[0]
    return {PRETTY.get(s["label"], s["label"]): float(s["score"]) for s in scores}


def _baseline_scores(text: str) -> dict:
    if _baseline is None:
        return {"model yok": 1.0}
    proba = _baseline.predict_proba([text])[0]
    return {PRETTY.get(int(c), str(c)): float(p)
            for c, p in zip(_baseline.classes_, proba)}


def analyze(text: str):
    """İki modelin de tahminini döndür (BERTurk, Baseline)."""
    text = (text or "").strip()
    if not text:
        empty = {"Metin girin": 1.0}
        return empty, empty
    return _berturk_scores(text), _baseline_scores(text)


EXAMPLES = [
    "Ürün çok kaliteli, kargo hızlıydı, kesinlikle tavsiye ederim.",
    "Paranıza yazık, iki gün sonra bozuldu. Kesinlikle almayın.",
    "Kötü diyemem, fiyatına göre gayet başarılı.",
    "Fena değil ama beklediğim performansı vermedi.",
    "Rezalet bir ürün, kutusu bile ezik geldi.",
    "İdare eder, ne iyi ne kötü.",
]

HEADER = """
<div align="center">

# 🇹🇷 Türkçe Ürün Yorumu Duygu Analizi

Bir ürün yorumu yazın; **fine-tune edilmiş BERTurk** modeli onu
**pozitif / negatif** olarak sınıflandırsın.
Yanında **baseline** (TF-IDF + Lojistik Regresyon) tahminini de görün —
transformer'ın farkı özellikle *olumsuzlama* ve *bağlam* içeren yorumlarda ortaya çıkar.

</div>
"""

FOOTER = f"""
---
<div align="center">

**BERTurk** macro-F1 = 0.86 · **Baseline** macro-F1 = 0.74 · aynı test kümesinde

[⭐ GitHub]({GITHUB_URL}) · [🤗 Model](https://huggingface.co/{BERTURK_MODEL_ID}) ·
Veri: <code>fthbrmnby/turkish_product_reviews</code>

</div>
"""

theme = gr.themes.Soft(primary_hue="red", secondary_hue="slate")

with gr.Blocks(title="Türkçe Duygu Analizi") as demo:
    gr.Markdown(HEADER)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(
                lines=5, label="Ürün yorumu",
                placeholder="Örn: Ürün çok kaliteli, herkese tavsiye ederim...",
            )
            with gr.Row():
                btn = gr.Button("🔍 Analiz Et", variant="primary")
                clear = gr.ClearButton(inp, value="Temizle")
        with gr.Column(scale=1):
            out_berturk = gr.Label(num_top_classes=2, label="🤖 BERTurk (ana model)")
            out_baseline = gr.Label(num_top_classes=2, label="📊 Baseline (kıyas)")

    gr.Examples(examples=EXAMPLES, inputs=inp, label="Örnek yorumlar (tıklayın)")
    gr.Markdown(FOOTER)

    btn.click(analyze, inputs=inp, outputs=[out_berturk, out_baseline])
    inp.submit(analyze, inputs=inp, outputs=[out_berturk, out_baseline])
    clear.add([out_berturk, out_baseline])

if __name__ == "__main__":
    demo.launch(theme=theme)
