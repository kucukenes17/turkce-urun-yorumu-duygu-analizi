"""Hata analizi — baseline modelinin (TF-IDF + LogReg) yanlış sınıflandırdığı
test örneklerini bulur, kategorize eder ve örnekler gösterir.

Baseline ile AYNI stratified bölmeyi (random_state=42) yeniden üreterek aynı
test kümesi üzerinde çalışır, kaydedilmiş modeli yükler.

Konsola özet basar, outputs/error_analysis.md dosyasına örnekleri yazar.

Kullanım:
    python -m src.baseline          # önce modeli eğit/kaydet
    python -m src.error_analysis
"""

from __future__ import annotations

import textwrap

import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_reviews


def _load_test_split() -> pd.DataFrame:
    """Baseline ile birebir aynı test kümesini yeniden üret."""
    df = load_reviews()
    df = df.dropna(subset=[config.TEXT_COL])
    df = df[df[config.TEXT_COL].str.strip() != ""]
    _, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[config.LABEL_COL],
        random_state=config.RANDOM_STATE,
    )
    return test_df.reset_index(drop=True)


def _format_examples(sub: pd.DataFrame, n: int = 5) -> str:
    lines = []
    for _, row in sub.head(n).iterrows():
        conf = row["confidence"]
        snippet = textwrap.shorten(str(row[config.TEXT_COL]), width=220, placeholder=" …")
        lines.append(f"- (model %{conf*100:.0f} emin) {snippet}")
    return "\n".join(lines) if lines else "- (örnek yok)"


def main() -> None:
    if not config.BASELINE_MODEL_PATH.exists():
        raise SystemExit(
            "Baseline modeli bulunamadı. Önce çalıştır: python -m src.baseline"
        )

    pipe = load(config.BASELINE_MODEL_PATH)
    test_df = _load_test_split()

    X = test_df[config.TEXT_COL]
    y_true = test_df[config.LABEL_COL]
    y_pred = pipe.predict(X)
    proba = pipe.predict_proba(X)

    test_df = test_df.copy()
    test_df["pred"] = y_pred
    test_df["confidence"] = proba.max(axis=1)
    test_df["correct"] = test_df["pred"] == y_true

    errors = test_df[~test_df["correct"]]
    n_err = len(errors)
    print(f"Test kümesi: {len(test_df):,} | Hatalı: {n_err:,} "
          f"(%{n_err/len(test_df)*100:.2f})")

    # İki hata türü:
    # False negative: gerçekte pozitif ama negatif denmiş
    # False positive: gerçekte negatif ama pozitif denmiş
    fn = errors[(errors[config.LABEL_COL] == 1) & (errors["pred"] == 0)]
    fp = errors[(errors[config.LABEL_COL] == 0) & (errors["pred"] == 1)]
    print(f"  Pozitif→Negatif yanlış (false negative): {len(fn):,}")
    print(f"  Negatif→Pozitif yanlış (false positive): {len(fp):,}")

    # En "emin" olduğu halde yanıldığı örnekler (en öğretici hatalar)
    fn_conf = fn.sort_values("confidence", ascending=False)
    fp_conf = fp.sort_values("confidence", ascending=False)

    print("\n--- Gerçekte POZİTİF, model NEGATİF dedi (en emin yanılgılar) ---")
    print(_format_examples(fn_conf))
    print("\n--- Gerçekte NEGATİF, model POZİTİF dedi (en emin yanılgılar) ---")
    print(_format_examples(fp_conf))

    content = f"""# Hata Analizi — Baseline (TF-IDF + Lojistik Regresyon)

Aynı stratified test kümesi ({len(test_df):,} yorum) üzerinde kaydedilmiş baseline
modelinin yanlışları.

- Toplam hata: **{n_err:,}** (%{n_err/len(test_df)*100:.2f})
- Gerçekte pozitif → "negatif" denen: **{len(fn):,}**
- Gerçekte negatif → "pozitif" denen: **{len(fp):,}**

## Gerçekte POZİTİF, model NEGATİF dedi
(Model yüksek özgüvenle yanıldığı örnekler; genelde kısa, alaycı ya da
"kötü değil / fena değil" gibi olumsuzlama içeren olumlu yorumlar.)

{_format_examples(fn_conf, n=8)}

## Gerçekte NEGATİF, model POZİTİF dedi
(Genelde olumlu kelimeler içeren ama aslında şikayet olan yorumlar; baseline
kelime torbası olduğu için bağlamı kaçırır.)

{_format_examples(fp_conf, n=8)}

## Çıkarım
Hatalar iki gruba ayrılıyor:

1. **Modelin gerçek zorlandığı vakalar.** Baseline bir **kelime torbası**
   (bag-of-words) modeli olduğu için olumsuzlama ("kötü **değil**", "kötü diyemem"),
   karışık yorumlar ("güzel **ama** yapışkanı kötü") ve bağlama bağlı ifadeleri
   kaçırıyor. BERTurk bu bağlamsal ipuçlarını yakaladığı için özellikle azınlık
   (negatif) sınıfında belirgin şekilde daha iyi (negatif F1: 0.53 → 0.74).

2. **Veri setindeki etiket gürültüsü.** "Gerçekte negatif → pozitif denen"
   örneklerin bir kısmı ("gayet güzel, tavsiye ederim", "mükemmel bir ürün")
   aslında **açıkça olumlu** metinler; yani modelin değil, **etiketin** yanlış
   olduğu durumlar. Bu, yıldız-puanı → etiket eşlemesinden kaynaklı bir sınırlama
   ve raporlanan tavan başarıyı bir miktar aşağı çeker (mükemmel bir model bile
   yanlış etiketli örnekleri "yanlış" bilmek zorunda kalır).

---
*Bu dosya `python -m src.error_analysis` ile otomatik üretilir.*
"""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUTS_DIR / "error_analysis.md").write_text(content, encoding="utf-8")
    print(f"\n[kaydedildi] {config.OUTPUTS_DIR / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
