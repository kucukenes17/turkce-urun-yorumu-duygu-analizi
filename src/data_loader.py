"""Türkçe ürün yorumu veri setini indir, önbelleğe al ve pandas DataFrame döndür.

Kullanım:
    from src.data_loader import load_reviews
    df = load_reviews()          # text, label, label_name sütunlu DataFrame

    python -m src.data_loader    # indirir, önbelleğe yazar, özet basar
"""

from __future__ import annotations

import pandas as pd

from src import config


def _download_from_hub() -> pd.DataFrame:
    """Hugging Face Hub'dan indir ve normalize edilmiş DataFrame döndür."""
    from datasets import load_dataset

    try:
        ds = load_dataset(config.DATASET_ID, split="train")
    except (ValueError, RuntimeError) as exc:
        # Bazı datasets sürümlerinde script tabanlı setler açık izin ister.
        print(f"[uyarı] Standart yükleme başarısız ({exc}); "
              "trust_remote_code=True ile tekrar deneniyor...")
        ds = load_dataset(config.DATASET_ID, split="train", trust_remote_code=True)

    df = ds.to_pandas()
    df = df.rename(columns={
        config.SRC_TEXT_COL: config.TEXT_COL,
        config.SRC_LABEL_COL: config.LABEL_COL,
    })
    df = df[[config.TEXT_COL, config.LABEL_COL]].copy()
    df[config.LABEL_NAME_COL] = df[config.LABEL_COL].map(config.LABEL_NAMES)
    return df


def load_reviews(use_cache: bool = True) -> pd.DataFrame:
    """Yorumları DataFrame olarak yükle (text, label, label_name).

    use_cache=True ise ve yerel parquet önbelleği varsa doğrudan onu okur;
    yoksa Hub'dan indirip önbelleğe yazar.
    """
    if use_cache and config.RAW_CACHE.exists():
        return pd.read_parquet(config.RAW_CACHE)

    df = _download_from_hub()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.RAW_CACHE, index=False)
    return df


def main() -> None:
    df = load_reviews()
    print(f"Veri seti: {config.DATASET_ID}")
    print(f"Toplam kayıt: {len(df):,}")
    print(f"Sütunlar: {list(df.columns)}")
    print(f"Önbellek: {config.RAW_CACHE}")
    print("\nİlk 5 kayıt:")
    with pd.option_context("display.max_colwidth", 80):
        print(df.head())


if __name__ == "__main__":
    main()
