"""Gradio demo — Türkçe ürün yorumu duygu sınıflandırıcı (premium arayüz).

İki modeli yan yana karşılaştırır (BERTurk vs baseline), özel tasarlanmış
modern bir arayüzle: koyu tema, gradient'ler, cam efekti kartlar, animasyonlu
güven çubukları.

Yerel çalıştırma:
    pip install -r requirements.txt
    python app.py            # tarayıcı: http://127.0.0.1:7860

Hugging Face Spaces: app.py + requirements.txt kök dizinde → doğrudan Space.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from joblib import load
from transformers import pipeline

BERTURK_MODEL_ID = "Eneskck/berturk-turkish-product-sentiment"
BASELINE_PATH = Path(__file__).parent / "models" / "baseline_tfidf_logreg.joblib"
GITHUB_URL = "https://github.com/kucukenes17/turkce-urun-yorumu-duygu-analizi"

# label -> (gösterilecek ad, "pos"/"neg")
POL = {
    "pozitif": ("Pozitif", "pos"), "negatif": ("Negatif", "neg"),
    "LABEL_1": ("Pozitif", "pos"), "LABEL_0": ("Negatif", "neg"),
    1: ("Pozitif", "pos"), 0: ("Negatif", "neg"),
}

_berturk = pipeline("text-classification", model=BERTURK_MODEL_ID, top_k=None)
_baseline = load(BASELINE_PATH) if BASELINE_PATH.exists() else None


def _top(scores: dict):
    """En yüksek skoru (ad, polarite, olasılık) olarak döndür."""
    label, prob = max(scores.items(), key=lambda kv: kv[1])
    name, pol = POL.get(label, (str(label), "neg"))
    return name, pol, prob


def _berturk_scores(text):
    return {s["label"]: float(s["score"]) for s in _berturk(text)[0]}


def _baseline_scores(text):
    if _baseline is None:
        return {"negatif": 1.0}
    proba = _baseline.predict_proba([text])[0]
    return {int(c): float(p) for c, p in zip(_baseline.classes_, proba)}


def _card(model_name, tag, tag_kind, scores):
    name, pol, prob = _top(scores)
    pct = round(prob * 100)
    emoji = "😊" if pol == "pos" else "😞"
    arrow = "👍" if pol == "pos" else "👎"
    return f"""
<div class="card {pol}">
  <div class="card-head">
    <span class="model">{model_name}</span>
    <span class="tag {tag_kind}">{tag}</span>
  </div>
  <div class="verdict">
    <span class="emoji">{emoji}</span>
    <span class="label">{name} {arrow}</span>
  </div>
  <div class="bar"><div class="fill" style="width:{pct}%"></div></div>
  <div class="pct">%{pct} <span>güven</span></div>
</div>"""


PLACEHOLDER = """
<div class="card empty">
  <div class="card-head"><span class="model">{m}</span><span class="tag {k}">{t}</span></div>
  <div class="verdict placeholder"><span class="emoji">💬</span>
  <span class="label">Bir yorum yazın</span></div>
  <div class="bar"><div class="fill" style="width:0%"></div></div>
  <div class="pct">—</div>
</div>"""

EMPTY_BERTURK = PLACEHOLDER.format(m="🤖 BERTurk", k="primary", t="ANA MODEL")
EMPTY_BASELINE = PLACEHOLDER.format(m="📊 Baseline", k="muted", t="KIYAS")


def analyze(text: str):
    text = (text or "").strip()
    if not text:
        return EMPTY_BERTURK, EMPTY_BASELINE
    b = _card("🤖 BERTurk", "ANA MODEL", "primary", _berturk_scores(text))
    base = _card("📊 Baseline", "KIYAS", "muted", _baseline_scores(text))
    return b, base


EXAMPLES = [
    "Ürün çok kaliteli, kargo hızlıydı, kesinlikle tavsiye ederim.",
    "Paranıza yazık, iki gün sonra bozuldu. Kesinlikle almayın.",
    "Kötü diyemem, fiyatına göre gayet başarılı.",
    "Fena değil ama beklediğim performansı vermedi.",
    "Rezalet bir ürün, kutusu bile ezik geldi.",
]

CSS = """
:root {
  --bg: #0b0b14; --card: rgba(255,255,255,.04); --stroke: rgba(255,255,255,.09);
  --txt: #e9e9f2; --muted: #9a9ab0; --accent: #7c5cff; --accent2: #b06cff;
  --pos: #22c55e; --pos2: #34d399; --neg: #ef4444; --neg2: #f87171;
}
.gradio-container, .gradio-container * { font-family: 'Segoe UI', Inter, system-ui, -apple-system, sans-serif !important; }
.gradio-container {
  max-width: 1000px !important; margin: 0 auto !important;
  background: radial-gradient(1100px 600px at 50% -10%, #1a1140 0%, var(--bg) 55%) !important;
  color: var(--txt) !important;
}
footer { display: none !important; }

/* Hero */
#hero { text-align: center; padding: 34px 10px 8px; }
#hero .kicker {
  display:inline-block; font-size:12px; letter-spacing:.18em; font-weight:600;
  color:var(--accent2); background:rgba(124,92,255,.12); border:1px solid rgba(124,92,255,.3);
  padding:6px 14px; border-radius:999px; margin-bottom:18px; text-transform:uppercase;
}
#hero h1 {
  font-size: 46px; font-weight: 800; line-height:1.12; margin:0 0 14px;
  color:#f4f0ff; letter-spacing:-.5px; text-shadow:0 2px 40px rgba(124,92,255,.35);
}
#hero h1 .grad { color: var(--accent2); }
#hero p { color: var(--muted); font-size:16px; max-width:560px; margin:0 auto; line-height:1.6; }

/* Panels / cards (Gradio bloklarını sadeleştir) */
.gr-group, .gr-form, .block, .gr-box { background: transparent !important; border: none !important; box-shadow:none !important; }
#input-card {
  background: var(--card) !important; border:1px solid var(--stroke) !important;
  border-radius: 20px !important; padding: 20px !important; backdrop-filter: blur(14px);
  box-shadow: 0 20px 60px rgba(0,0,0,.45);
}
textarea {
  background: rgba(0,0,0,.25) !important; color: var(--txt) !important;
  border:1px solid var(--stroke) !important; border-radius:14px !important;
  font-size:16px !important; padding:14px !important;
}
textarea:focus { border-color: var(--accent) !important; box-shadow:0 0 0 3px rgba(124,92,255,.25) !important; }
label span { color: var(--muted) !important; font-weight:600 !important; }

/* Buttons */
button.primary, #go {
  background: linear-gradient(100deg, var(--accent), var(--accent2)) !important;
  border:none !important; color:#fff !important; font-weight:700 !important;
  border-radius:14px !important; padding:12px 20px !important; font-size:15px !important;
  box-shadow: 0 8px 24px rgba(124,92,255,.4); transition: transform .12s ease, box-shadow .12s ease;
}
#go:hover { transform: translateY(-2px); box-shadow:0 12px 32px rgba(124,92,255,.55) !important; }
#clear { background: rgba(255,255,255,.06) !important; color: var(--muted) !important;
  border:1px solid var(--stroke) !important; border-radius:14px !important; font-weight:600 !important; }

/* Result cards */
.card {
  background: var(--card); border:1px solid var(--stroke); border-radius:20px;
  padding:22px; backdrop-filter: blur(14px); position:relative; overflow:hidden;
  box-shadow: 0 16px 44px rgba(0,0,0,.4); min-height:172px;
}
.card::before { content:""; position:absolute; inset:0 0 auto 0; height:3px; opacity:.9; }
.card.pos::before { background: linear-gradient(90deg,var(--pos),var(--pos2)); }
.card.neg::before { background: linear-gradient(90deg,var(--neg),var(--neg2)); }
.card.empty::before { background: var(--stroke); }
.card-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
.card .model { font-weight:700; font-size:16px; color:var(--txt); }
.tag { font-size:10px; letter-spacing:.12em; font-weight:700; padding:4px 10px; border-radius:999px; }
.tag.primary { color:#c9b8ff; background:rgba(124,92,255,.18); border:1px solid rgba(124,92,255,.4); }
.tag.muted { color:var(--muted); background:rgba(255,255,255,.05); border:1px solid var(--stroke); }
.verdict { display:flex; align-items:center; gap:12px; margin-bottom:16px; }
.verdict .emoji { font-size:38px; line-height:1; }
.verdict .label { font-size:30px; font-weight:800; }
.card.pos .label { color:var(--pos2); }
.card.neg .label { color:var(--neg2); }
.verdict.placeholder .label { color:var(--muted); font-weight:600; font-size:22px; }
.bar { height:10px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; }
.fill { height:100%; border-radius:999px; transition: width .7s cubic-bezier(.22,1,.36,1); }
.card.pos .fill { background: linear-gradient(90deg,var(--pos),var(--pos2)); }
.card.neg .fill { background: linear-gradient(90deg,var(--neg),var(--neg2)); }
.card.empty .fill { background: var(--stroke); }
.pct { margin-top:10px; font-size:22px; font-weight:800; color:var(--txt); }
.pct span { font-size:13px; font-weight:500; color:var(--muted); }

/* Examples as chips */
#examples { margin-top:6px; }
.gr-samples-table, #examples table { background:transparent !important; border:none !important; }
#examples button, .gr-sample-textbox {
  background: rgba(255,255,255,.05) !important; border:1px solid var(--stroke) !important;
  color: var(--muted) !important; border-radius:999px !important; font-size:13px !important;
  padding:8px 14px !important; transition: all .15s ease;
}
#examples button:hover { border-color: var(--accent) !important; color:#fff !important; background:rgba(124,92,255,.14) !important; }

/* Footer */
#foot { text-align:center; color:var(--muted); font-size:13px; padding:22px 10px 30px; }
#foot a { color:var(--accent2); text-decoration:none; font-weight:600; }
#foot a:hover { text-decoration:underline; }
#foot .metrics { margin-bottom:10px; }
#foot .metrics b { color:var(--txt); }
"""

HERO = """
<div id="hero">
  <span class="kicker">NLP · Duygu Analizi</span>
  <h1>Türkçe Ürün Yorumu<br><span class="grad">Duygu Analizi</span></h1>
  <p>Bir ürün yorumu yazın; fine-tune edilmiş <b>BERTurk</b> onu saniyeler içinde
  sınıflandırsın. Yanında klasik <b>baseline</b>'ı da görün — farkı özellikle
  olumsuzlama ve bağlam içeren yorumlarda ortaya çıkar.</p>
</div>
"""

FOOT = f"""
<div id="foot">
  <div class="metrics"><b>BERTurk</b> macro-F1 0.86 &nbsp;·&nbsp; <b>Baseline</b> macro-F1 0.74
  &nbsp;·&nbsp; 235K yorum, aynı test kümesi</div>
  <a href="{GITHUB_URL}" target="_blank">⭐ GitHub</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/{BERTURK_MODEL_ID}" target="_blank">🤗 Model</a> &nbsp;·&nbsp;
  <span>Veri: fthbrmnby/turkish_product_reviews</span>
</div>
"""

theme = gr.themes.Base(primary_hue="purple", neutral_hue="slate")

with gr.Blocks(css=CSS, theme=theme, title="Türkçe Duygu Analizi") as demo:
    gr.HTML(HERO)

    with gr.Group(elem_id="input-card"):
        inp = gr.Textbox(
            lines=4, label="Ürün yorumu", show_label=True,
            placeholder="Örn: Ürün beklentimin çok üzerinde, herkese tavsiye ederim...",
        )
        with gr.Row():
            go = gr.Button("✨ Analiz Et", elem_id="go", scale=3)
            clear = gr.Button("Temizle", elem_id="clear", scale=1)

    with gr.Row():
        out_b = gr.HTML(EMPTY_BERTURK)
        out_base = gr.HTML(EMPTY_BASELINE)

    gr.Examples(examples=EXAMPLES, inputs=inp, elem_id="examples",
                label="Hızlı dene:")
    gr.HTML(FOOT)

    go.click(analyze, inputs=inp, outputs=[out_b, out_base])
    inp.submit(analyze, inputs=inp, outputs=[out_b, out_base])
    clear.click(lambda: ("", EMPTY_BERTURK, EMPTY_BASELINE),
                outputs=[inp, out_b, out_base])

if __name__ == "__main__":
    demo.launch(theme=theme)
