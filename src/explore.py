"""Keşifçi veri analizi (EDA) — Türkçe ürün yorumu veri seti.

Konsola özet basar, outputs/figures/ altına grafik(ler) kaydeder ve
outputs/eda_summary.md dosyasına metinsel bir özet yazar.

Kullanım:
    python -m src.explore
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")  # başsız (headless) ortamda dosyaya çizim için
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.data_loader import load_reviews

sns.set_theme(style="whitegrid")


def _add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Karakter ve kelime uzunluğu sütunları ekle."""
    df = df.copy()
    df["char_len"] = df[config.TEXT_COL].str.len()
    df["word_len"] = df[config.TEXT_COL].str.split().str.len()
    return df


def _class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[config.LABEL_NAME_COL].value_counts()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"adet": counts, "yuzde": pct})


def _plot_class_distribution(dist: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=dist.index, y=dist["adet"], ax=ax, hue=dist.index, legend=False)
    ax.set_title("Sınıf Dağılımı (ham veri)")
    ax.set_xlabel("Etiket")
    ax.set_ylabel("Yorum adedi")
    for i, (adet, yuzde) in enumerate(zip(dist["adet"], dist["yuzde"])):
        ax.text(i, adet, f"{adet:,}\n(%{yuzde})", ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.15)
    fig.tight_layout()
    path = config.FIGURES_DIR / "class_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[kaydedildi] {path}")


def _plot_length_hist(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 99. persentil ile kırparak uzun kuyruğun grafiği bozmasını engelle.
    for ax, col, title in (
        (axes[0], "word_len", "Kelime sayısı"),
        (axes[1], "char_len", "Karakter sayısı"),
    ):
        upper = df[col].quantile(0.99)
        sns.histplot(
            data=df[df[col] <= upper],
            x=col, hue=config.LABEL_NAME_COL, bins=50,
            element="step", stat="density", common_norm=False, ax=ax,
        )
        ax.set_title(f"{title} dağılımı (≤ %99 persentil)")
        ax.set_xlabel(title)
    fig.tight_layout()
    path = config.FIGURES_DIR / "length_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[kaydedildi] {path}")


def _length_stats(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(config.LABEL_NAME_COL)[["char_len", "word_len"]].agg(
        ["mean", "median", "min", "max"]
    )
    overall = df[["char_len", "word_len"]].agg(["mean", "median", "min", "max"]).T
    overall.index = pd.MultiIndex.from_product([["tümü"], overall.index])
    return agg.round(1), overall.round(1)


def _sample_reviews(df: pd.DataFrame, n: int = 3) -> str:
    lines = []
    for label_val, name in config.LABEL_NAMES.items():
        sub = df[df[config.LABEL_COL] == label_val]
        if sub.empty:
            continue
        lines.append(f"\n### {name.capitalize()} örnekler")
        for txt in sub[config.TEXT_COL].head(n):
            snippet = textwrap.shorten(str(txt), width=200, placeholder=" …")
            lines.append(f"- {snippet}")
    return "\n".join(lines)


def _write_summary(df: pd.DataFrame, dist: pd.DataFrame,
                   dupes: int, nulls: int) -> None:
    maj = dist["yuzde"].max()
    summary = f"""# EDA Özeti — Türkçe Ürün Yorumu Veri Seti

**Veri seti:** `{config.DATASET_ID}`

## Genel
- Toplam kayıt: **{len(df):,}**
- Sütunlar: `{config.TEXT_COL}`, `{config.LABEL_COL}`, `{config.LABEL_NAME_COL}`
- Tekrar eden yorum sayısı: **{dupes:,}**
- Boş/eksik metin sayısı: **{nulls:,}**

## Sınıf Dağılımı
{dist.to_markdown()}

> ⚠️ **Veri seti dengesiz:** çoğunluk sınıfı ~%{maj:.1f}. Bu nedenle ölçüt olarak
> accuracy yanıltıcıdır; **F1 (macro / pozitif-negatif ayrı)** ve confusion matrix
> raporlanmalı. Eğitim/test bölmesi **stratified** yapılmalı ve modelde
> **class weight** veya örnekleme dengelenmesi değerlendirilmeli.

## Metin Uzunluğu (kelime / karakter)
Grafikler: `outputs/figures/length_distribution.png`

## Örnek Yorumlar
{_sample_reviews(df)}

---
*Bu dosya `python -m src.explore` ile otomatik üretilir.*
"""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EDA_SUMMARY.write_text(summary, encoding="utf-8")
    print(f"[kaydedildi] {config.EDA_SUMMARY}")


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Veri yükleniyor...")
    df = load_reviews()
    df = _add_length_columns(df)

    print(f"\n=== {config.DATASET_ID} ===")
    print(f"Toplam kayıt: {len(df):,}")
    print(f"Sütun tipleri:\n{df[[config.TEXT_COL, config.LABEL_COL]].dtypes}")

    # Sınıf dağılımı
    dist = _class_distribution(df)
    print("\n--- Sınıf dağılımı ---")
    print(dist.to_string())
    print(f"\n⚠️  Çoğunluk sınıfı ~%{dist['yuzde'].max():.1f} → veri seti DENGESİZ. "
          "Ölçüt olarak F1 + confusion matrix kullan (accuracy değil).")

    # Uzunluk istatistikleri
    per_class, overall = _length_stats(df)
    print("\n--- Uzunluk istatistikleri (sınıf bazında) ---")
    print(per_class.to_string())
    print("\n--- Uzunluk istatistikleri (tümü) ---")
    print(overall.to_string())

    # Kalite kontrolleri
    dupes = int(df[config.TEXT_COL].duplicated().sum())
    nulls = int(df[config.TEXT_COL].isna().sum()
                + (df[config.TEXT_COL].str.strip() == "").sum())
    print(f"\nTekrar eden yorum: {dupes:,} | Boş/eksik metin: {nulls:,}")

    # Örnekler
    print("\n--- Örnek yorumlar ---")
    print(_sample_reviews(df))

    # Grafikler + özet rapor
    print()
    _plot_class_distribution(dist)
    _plot_length_hist(df)
    _write_summary(df, dist, dupes, nulls)

    print("\nEDA tamamlandı.")


if __name__ == "__main__":
    main()
