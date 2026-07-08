# EDA Özeti — Türkçe Ürün Yorumu Veri Seti

**Veri seti:** `fthbrmnby/turkish_product_reviews`

## Genel
- Toplam kayıt: **235,165**
- Sütunlar: `text`, `label`, `label_name`
- Tekrar eden yorum sayısı: **1,657**
- Boş/eksik metin sayısı: **21**

## Sınıf Dağılımı
| label_name   |   adet |   yuzde |
|:-------------|-------:|--------:|
| pozitif      | 220284 |   93.67 |
| negatif      |  14881 |    6.33 |

> ⚠️ **Veri seti dengesiz:** çoğunluk sınıfı ~%93.7. Bu nedenle ölçüt olarak
> accuracy yanıltıcıdır; **F1 (macro / pozitif-negatif ayrı)** ve confusion matrix
> raporlanmalı. Eğitim/test bölmesi **stratified** yapılmalı ve modelde
> **class weight** veya örnekleme dengelenmesi değerlendirilmeli.

## Metin Uzunluğu (kelime / karakter)
Grafikler: `outputs/figures/length_distribution.png`

## Örnek Yorumlar

### Negatif örnekler
- beklentimin altında bir ürün kaliteli değil
- 3. kademe hız da motor titreme yapıyor.bu sebebten 3.kademe kullanımda uzun ömürlü olacağını zannetmiyorum.elide rahatsız ediyor.
- başlığı sabit durmuyor. arka koruma demiri üfleme hızını 2.sewiye yapınca yüksek ses cikariyor

### Pozitif örnekler
- fena değil paraya göre iyi.
- ürün kaliteli ve çok kullanışlı kargo ya verilmesi ve elime çabuk ulaşması da gayet başarılı idi
- fiyat-performans karşılaştırması yaptığınızda gayet iyi bir ürün. en önemlisi ısınma yapmıyor.

---
*Bu dosya `python -m src.explore` ile otomatik üretilir.*
