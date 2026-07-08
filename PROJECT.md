# Türkçe Ürün Yorumu Duygu Sınıflandırıcı

## Amaç
Türkçe ürün yorumlarını pozitif/negatif olarak sınıflandıran bir NLP projesi.
Portfolyo/CV için: baseline → transformer karşılaştırması + dürüst hata analizi.

## Yaklaşım
- Veri: Hugging Face'ten hazır Türkçe sentiment seti (yıldız puanı = etiket)
- Baseline: TF-IDF + Lojistik Regresyon (scikit-learn)
- Ana model: BERTurk (dbmdz/bert-base-turkish-cased) fine-tuning
- Değerlendirme: accuracy, F1, confusion matrix, hata analizi

## Yol Haritası
- Hafta 1: Veri yükleme + keşif + baseline model
- Hafta 2: BERTurk fine-tuning (Colab GPU)
- Hafta 3: Değerlendirme + hata analizi + README + demo

## Prensipler
- Baseline'sız derin modele geçme (kıyas noktası şart)
- README'de sınırlamalar ve hata örnekleri açıkça belirtilecek
- requirements.txt + tek komutla çalıştırma

## Stretch (opsiyonel)
- Aspect-based sentiment ("neyin hakkında pozitif/negatif")
