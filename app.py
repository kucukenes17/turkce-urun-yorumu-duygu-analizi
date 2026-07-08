"""Gradio demo — Türkçe ürün yorumu duygu sınıflandırıcı.

Arayüz, Anthropic frontend-design ilkelerine göre tasarlandı: konuya demirlenmiş
(ürün yorumu + ML kıyası), editoryal/aydınlık bir "Yorum Laboratuvarı" estetiği.
Sinyal öğe: iki model AYRIŞINCA öne çıkan uyarı — projenin tezi budur.

Tipografi: Space Grotesk (başlık), Inter (metin), Space Mono (skor/metrik).

Yerel çalıştırma:
    pip install -r requirements.txt
    python app.py            # tarayıcı: http://127.0.0.1:7860
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from joblib import load
from transformers import pipeline

BERTURK_MODEL_ID = "Eneskck/berturk-turkish-product-sentiment"
BASELINE_PATH = Path(__file__).parent / "models" / "baseline_tfidf_logreg.joblib"
GITHUB_URL = "https://github.com/kucukenes17/turkce-urun-yorumu-duygu-analizi"

POL = {
    "pozitif": ("Pozitif", "pos"), "negatif": ("Negatif", "neg"),
    "LABEL_1": ("Pozitif", "pos"), "LABEL_0": ("Negatif", "neg"),
    1: ("Pozitif", "pos"), 0: ("Negatif", "neg"),
}

_berturk = pipeline("text-classification", model=BERTURK_MODEL_ID, top_k=None)
_baseline = load(BASELINE_PATH) if BASELINE_PATH.exists() else None


def _top(scores: dict):
    label, prob = max(scores.items(), key=lambda kv: kv[1])
    name, pol = POL.get(label, (str(label), "neg"))
    return name, pol, round(prob * 100)


def _berturk_scores(text):
    return {s["label"]: float(s["score"]) for s in _berturk(text)[0]}


def _baseline_scores(text):
    if _baseline is None:
        return {"negatif": 1.0}
    proba = _baseline.predict_proba([text])[0]
    return {int(c): float(p) for c, p in zip(_baseline.classes_, proba)}


def _card(model, role, f1, scores, star=False):
    name, pol, pct = _top(scores)
    tag = '<span class="star">★ ana model</span>' if star else ""
    return f"""
<article class="verdict {pol}">
  <div class="model">{model}{tag}</div>
  <div class="role">{role}</div>
  <div class="mark">{name}</div>
  <div class="meter"><i style="width:{pct}%"></i></div>
  <div class="score"><b>%{pct}</b> güven · macro-F1 {f1}</div>
</article>"""


def _placeholder(model, role, f1, star=False):
    tag = '<span class="star">★ ana model</span>' if star else ""
    return f"""
<article class="verdict idle">
  <div class="model">{model}{tag}</div>
  <div class="role">{role}</div>
  <div class="mark idle">—</div>
  <div class="meter"><i style="width:0%"></i></div>
  <div class="score">yorum bekleniyor · macro-F1 {f1}</div>
</article>"""


EMPTY_B = _placeholder("BERTurk", "fine-tune · transformer", "0.86", star=True)
EMPTY_BASE = _placeholder("Baseline", "TF-IDF + regresyon", "0.74")
EMPTY_BANNER = ""


def _banner(pol_b, pol_base):
    if pol_b == pol_base:
        return '<div class="verdictbar agree">İki model de aynı kararda.</div>'
    return ('<div class="verdictbar clash">Modeller ayrıştı — '
            'olumsuzlama ve bağlamın farkı tam burada görünür.</div>')


def analyze(text: str):
    text = (text or "").strip()
    if not text:
        return EMPTY_BANNER, EMPTY_B, EMPTY_BASE
    sb, sbase = _berturk_scores(text), _baseline_scores(text)
    _, pol_b, _ = _top(sb)
    _, pol_base, _ = _top(sbase)
    banner = _banner(pol_b, pol_base)
    return (banner,
            _card("BERTurk", "fine-tune · transformer", "0.86", sb, star=True),
            _card("Baseline", "TF-IDF + regresyon", "0.74", sbase))


EXAMPLES = [
    "Ürün çok kaliteli, kargo hızlıydı, kesinlikle tavsiye ederim.",
    "Paranıza yazık, iki gün sonra bozuldu. Kesinlikle almayın.",
    "Kötü diyemem, fiyatına göre gayet başarılı.",
    "Fena değil ama beklediğim performansı vermedi.",
    "Rezalet bir ürün, kutusu bile ezik geldi.",
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
  --paper:#FBFAF6; --panel:#FFFFFF; --ink:#1A1917; --muted:#77736A; --line:#E7E3D9;
  --pos:#14794E; --pos-soft:#E9F3ED; --neg:#C23A2B; --neg-soft:#FAEBE7;
  --grotesk:'Space Grotesk',system-ui,sans-serif; --body:'Inter',system-ui,sans-serif; --mono:'Space Mono',monospace;
}
.gradio-container, .gradio-container.dark {
  max-width:1400px !important; width:100% !important; margin:0 auto !important;
  padding:0 40px !important; color-scheme:light;
  background:var(--paper) !important; color:var(--ink) !important; font-family:var(--body) !important;
  /* Gradio tema degiskenlerini aydinliga zorla (dark modu ez) */
  --body-background-fill:var(--paper); --background-fill-primary:var(--paper);
  --background-fill-secondary:var(--paper); --block-background-fill:var(--panel);
  --block-label-background-fill:var(--panel); --block-border-color:var(--line);
  --border-color-primary:var(--line); --border-color-accent:var(--line);
  --input-background-fill:#FCFBF9; --input-background-fill-focus:#FCFBF9;
  --input-border-color:var(--line); --input-border-color-focus:var(--ink);
  --button-secondary-background-fill:transparent; --button-secondary-background-fill-hover:transparent;
  --button-secondary-text-color:var(--muted); --button-secondary-border-color:var(--line);
  --block-label-text-color:var(--muted); --body-text-color:var(--ink);
  --block-info-text-color:var(--muted);
}
.gradio-container * { font-family:var(--body); }
footer { display:none !important; }
.gr-group,.gr-form,.block,.gr-box,.gr-panel { background:transparent !important; border:none !important; box-shadow:none !important; }

/* Masthead / hero */
#hero { padding:46px 4px 10px; }
#hero .kicker { font-family:var(--mono); font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); }
#hero h1 { font-family:var(--grotesk); font-weight:700; font-size:52px; line-height:1.02;
  letter-spacing:-.02em; color:var(--ink); margin:14px 0 16px; }
#hero .lead { font-size:17px; line-height:1.65; color:var(--muted); max-width:600px; }
#hero .lead b { color:var(--ink); font-weight:600; }

/* Input */
#input-card { background:var(--panel) !important; border:1px solid var(--line) !important;
  border-radius:14px !important; padding:20px !important; }
label span { color:var(--muted) !important; font-family:var(--mono) !important; font-size:12px !important;
  letter-spacing:.06em !important; text-transform:uppercase !important; font-weight:400 !important; }
textarea { background:#FCFBF9 !important; color:var(--ink) !important; border:1px solid var(--line) !important;
  border-radius:10px !important; font-size:16px !important; font-family:var(--body) !important; padding:14px !important; }
textarea:focus { border-color:var(--ink) !important; box-shadow:none !important; }
#go { background:var(--ink) !important; color:#FBFAF6 !important; border:none !important; border-radius:10px !important;
  font-family:var(--grotesk) !important; font-weight:600 !important; font-size:15px !important; padding:12px 22px !important;
  transition:opacity .15s ease; }
#go:hover { opacity:.86; }
#clear { background:transparent !important; color:var(--muted) !important; border:1px solid var(--line) !important;
  border-radius:10px !important; font-weight:500 !important; }

/* Verdict bar (signature: disagreement = the thesis) */
.verdictbar { font-family:var(--mono); font-size:13px; letter-spacing:.02em; padding:12px 16px; border-radius:10px;
  margin:20px 0 4px; display:flex; align-items:center; gap:10px; animation:rise .5s ease both; }
.verdictbar::before { content:""; width:9px; height:9px; border-radius:50%; }
.verdictbar.agree { background:var(--pos-soft); color:var(--pos); }
.verdictbar.agree::before { background:var(--pos); }
.verdictbar.clash { background:#FBF3E4; color:#8A5B12; }
.verdictbar.clash::before { background:#C8891F; }

/* Verdict cards */
.verdict { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:26px 26px 24px; height:100%; }
.verdict .model { font-family:var(--grotesk); font-weight:700; font-size:17px; color:var(--ink); }
.verdict .star { font-family:var(--mono); font-size:10px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); margin-left:8px; white-space:nowrap; }
.verdict .role { font-family:var(--mono); font-size:11px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); margin:3px 0 22px; }
.verdict .mark { display:inline-block; font-family:var(--grotesk); font-weight:700; font-size:30px;
  letter-spacing:.01em; padding:7px 18px; border:2.5px solid currentColor; border-radius:9px; transform:rotate(-2deg); }
.verdict.pos .mark { color:var(--pos); background:var(--pos-soft); }
.verdict.neg .mark { color:var(--neg); background:var(--neg-soft); }
.verdict .mark.idle { color:var(--line); transform:none; }
.verdict .meter { height:6px; background:#EEEBE3; border-radius:99px; margin:22px 0 12px; overflow:hidden; }
.verdict .meter i { display:block; height:100%; border-radius:99px; transition:width .8s cubic-bezier(.22,1,.36,1); }
.verdict.pos .meter i { background:var(--pos); }
.verdict.neg .meter i { background:var(--neg); }
.verdict.idle .meter i { background:var(--line); }
.verdict .score { font-family:var(--mono); font-size:12px; color:var(--muted); }
.verdict .score b { font-size:15px; color:var(--ink); }

/* Examples */
#examples label span { color:var(--muted) !important; }
#examples button { background:var(--panel) !important; border:1px solid var(--line) !important; color:var(--muted) !important;
  border-radius:99px !important; font-size:13px !important; padding:8px 15px !important; transition:all .15s ease; }
#examples button:hover { border-color:var(--ink) !important; color:var(--ink) !important; }

/* Footer */
#foot { padding:26px 4px 34px; color:var(--muted); font-family:var(--mono); font-size:12px; letter-spacing:.02em; }
#foot .rule { height:1px; background:var(--line); margin-bottom:16px; }
#foot a { color:var(--ink); text-decoration:none; border-bottom:1px solid var(--line); }
#foot a:hover { border-color:var(--ink); }
#foot b { color:var(--ink); }

@keyframes rise { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
@media (prefers-reduced-motion: reduce) { .verdict .meter i, .verdictbar { transition:none; animation:none; } }
"""

HERO = """
<div id="hero">
  <div class="kicker">BERTurk × Baseline · Duygu Analizi</div>
  <h1>İki model, tek yorum.</h1>
  <p class="lead">Fine-tune edilmiş <b>BERTurk</b> ile klasik <b>TF-IDF baseline</b>'ı
  aynı yoruma bakarken izleyin. Çoğu yorumda aynı kararı verirler; olumsuzlama ve
  bağlam devreye girince ayrışırlar — bu projenin bütün meselesi o farkta.</p>
</div>
"""

FOOT = f"""
<div id="foot">
  <div class="rule"></div>
  <b>BERTurk</b> macro-F1 0.86 &nbsp; / &nbsp; <b>Baseline</b> macro-F1 0.74 &nbsp; / &nbsp;
  235K yorum · aynı test kümesi<br><br>
  <a href="{GITHUB_URL}" target="_blank">GitHub</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/{BERTURK_MODEL_ID}" target="_blank">Hugging Face modeli</a> &nbsp;·&nbsp;
  veri: fthbrmnby/turkish_product_reviews
</div>
"""

theme = gr.themes.Base(primary_hue="gray", neutral_hue="gray")

with gr.Blocks(css=CSS, theme=theme, title="Türkçe Duygu Analizi") as demo:
    gr.HTML(HERO)

    with gr.Row(equal_height=False, elem_id="workspace"):
        with gr.Column(scale=5, min_width=340):
            with gr.Group(elem_id="input-card"):
                inp = gr.Textbox(lines=6, label="Ürün yorumu",
                                 placeholder="Örn: Kötü diyemem, fiyatına göre gayet iyi.")
                with gr.Row():
                    go = gr.Button("Analiz et", elem_id="go", scale=3)
                    clear = gr.Button("Temizle", elem_id="clear", scale=1)
            gr.Examples(examples=EXAMPLES, inputs=inp, elem_id="examples",
                        label="Hızlı dene")
        with gr.Column(scale=7, min_width=420):
            banner = gr.HTML(EMPTY_BANNER)
            with gr.Row(equal_height=True):
                out_b = gr.HTML(EMPTY_B)
                out_base = gr.HTML(EMPTY_BASE)

    gr.HTML(FOOT)

    go.click(analyze, inputs=inp, outputs=[banner, out_b, out_base])
    inp.submit(analyze, inputs=inp, outputs=[banner, out_b, out_base])
    clear.click(lambda: ("", EMPTY_BANNER, EMPTY_B, EMPTY_BASE),
                outputs=[inp, banner, out_b, out_base])

if __name__ == "__main__":
    demo.launch(theme=theme)
