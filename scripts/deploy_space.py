"""Demoyu Hugging Face Spaces'e yayınlar ve model kartını Hub'a yükler.

ÖN KOŞUL — bir kez giriş yap (terminalde, token'ı SOHBETE değil oraya yapıştır):
    huggingface-cli login        # https://huggingface.co/settings/tokens (Write)

Sonra:
    python scripts/deploy_space.py

Yaptıkları:
  1) <kullanıcı>/yorumetre adında bir Gradio Space oluşturur (varsa günceller)
     ve app.py + requirements.txt + baseline modelini + Space README'sini yükler.
  2) MODEL_CARD.md'yi Eneskck/berturk-turkish-product-sentiment modeline README olarak yükler.
"""

from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "Eneskck/berturk-turkish-product-sentiment"
SPACE_NAME = "yorumetre"

SPACE_README = """---
title: Yorumetre
emoji: "◆"
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
---

# Yorumetre — Türkçe Ürün Yorumu Duygu Analizi

Fine-tune edilmiş BERTurk ile klasik TF-IDF baseline'ı yan yana karşılaştıran demo.
Kaynak: https://github.com/kucukenes17/turkce-urun-yorumu-duygu-analizi
"""


def main():
    api = HfApi()
    user = api.whoami()["name"]
    space_id = f"{user}/{SPACE_NAME}"
    print(f"Kullanıcı: {user} · Space: {space_id}")

    # 1) Space oluştur (varsa dokunma)
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    # Space README'sini yaz (frontmatter Space'i yapılandırır)
    (ROOT / "_space_readme.md").write_text(SPACE_README, encoding="utf-8")

    uploads = [
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
        ("models/baseline_tfidf_logreg.joblib", "models/baseline_tfidf_logreg.joblib"),
        ("_space_readme.md", "README.md"),
    ]
    for local, remote in uploads:
        api.upload_file(path_or_fileobj=str(ROOT / local), path_in_repo=remote,
                        repo_id=space_id, repo_type="space")
        print(f"  yüklendi → {remote}")

    (ROOT / "_space_readme.md").unlink(missing_ok=True)
    print(f"✓ Space hazır: https://huggingface.co/spaces/{space_id}")

    # 2) Model kartını yükle
    api.upload_file(path_or_fileobj=str(ROOT / "MODEL_CARD.md"), path_in_repo="README.md",
                    repo_id=MODEL_REPO, repo_type="model")
    print(f"✓ Model kartı yüklendi: https://huggingface.co/{MODEL_REPO}")


if __name__ == "__main__":
    main()
