"""Hızlı duman (smoke) testleri — ağır bağımlılık (torch/transformers) veya ağ gerektirmez.

Çalıştırma:
    pip install -r requirements-dev.txt
    pytest
"""

from pathlib import Path

import pytest

from src import config


def test_label_mapping():
    assert config.LABEL_NAMES == {0: "negatif", 1: "pozitif"}
    assert config.TEXT_COL == "text" and config.LABEL_COL == "label"


def test_paths_resolve():
    assert config.ROOT_DIR.exists()
    assert config.DATASET_ID == "fthbrmnby/turkish_product_reviews"


@pytest.mark.skipif(not config.BASELINE_MODEL_PATH.exists(),
                    reason="baseline modeli yok; önce: python -m src.baseline")
def test_baseline_predictions():
    """Kaydedilmiş baseline açık örneklerde doğru kutbu vermeli."""
    from joblib import load

    pipe = load(config.BASELINE_MODEL_PATH)
    texts = ["Bu ürün harika, çok memnunum, tavsiye ederim",
             "Berbat bir ürün, param çöpe gitti, almayın"]
    preds = pipe.predict(texts)
    assert config.LABEL_NAMES[int(preds[0])] == "pozitif"
    assert config.LABEL_NAMES[int(preds[1])] == "negatif"
    # olasılıklar iki sınıf için de dönmeli
    proba = pipe.predict_proba(texts)
    assert proba.shape == (2, 2)


def test_repo_layout():
    root = config.ROOT_DIR
    for f in ["app.py", "requirements.txt", "README.md", "LICENSE"]:
        assert (root / f).exists(), f"eksik: {f}"
