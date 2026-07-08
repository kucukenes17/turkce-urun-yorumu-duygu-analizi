# Hata Analizi — Baseline (TF-IDF + Lojistik Regresyon)

Aynı stratified test kümesi (47,029 yorum) üzerinde kaydedilmiş baseline
modelinin yanlışları.

- Toplam hata: **4,297** (%9.14)
- Gerçekte pozitif → "negatif" denen: **3,709**
- Gerçekte negatif → "pozitif" denen: **588**

## Gerçekte POZİTİF, model NEGATİF dedi
(Model yüksek özgüvenle yanıldığı örnekler; genelde kısa, alaycı ya da
"kötü değil / fena değil" gibi olumsuzlama içeren olumlu yorumlar.)

- (model %100 emin) kötü diyemem.
- (model %100 emin) veri taşıma hızı güzel ama aç kapa mekanizması çok kötü kullanışlı değil, tak çıkar yaparken kullanımı zor...
- (model %100 emin) bu sandalyeden iki adet çocuklarım için aldım. aldığıma pişman oldum. sırt yaslama kısmı rezalet, insanın boynunun ağrımasına sebep oluyor. yaslanma kısmı sağlam değil, oynayıp duruyor. komşuda almıştı onunkinin …
- (model %100 emin) bozuldu
- (model %100 emin) ürün çok güzel ve havalı görünüyor ama yapışkanı çok kötü yani çalışma masasının altına yapıştırmayı tavsiye etmem
- (model %100 emin) kimseye tavsiye etmiyorum :)) uzak durun. parfum budur gerisi teferruat..doganin gercek esintilerini bulabilecegininz kalici bir parfum. egzotik bir etkisi oldugu kesin.
- (model %99 emin) kulaklığının olmadığını biliyordum ama, ram performans ı diğer telefonlara nazaran kötü. ram hızı yüksek fakat, kullanılan uygulamalar yeniden açılırken nispeten yavaş açılıyor. ram management yöntemlerini denedim …
- (model %99 emin) oldukça ufak ürün eğer ufak şeyleri kaybederim derseniz tavsiye etmiyorum

## Gerçekte NEGATİF, model POZİTİF dedi
(Genelde olumlu kelimeler içeren ama aslında şikayet olan yorumlar; baseline
kelime torbası olduğu için bağlamı kaçırır.)

- (model %100 emin) gayet güzel ,şık ve kullanışlı.memnunum almak isteyenlere tavsiye ederim.
- (model %100 emin) uygun ve kaliteli tavsiye ederim
- (model %100 emin) mükemmel bir ürün özellikle ilköğretimde çocuğunuz varsa kaçırmayın.renkli ve kurşunkalaemler için çok iyi.mükemmel kalem açıyor.fiyatıda oldukça iyi
- (model %100 emin) güzel ürün hıziı kargo teşekkürler
- (model %100 emin) teşekkürler
- (model %100 emin) ürün beklediğimden iyi çıktı. severek kullanıyorum. sipariş sorunsuz elime ulaştı. tşk....
- (model %100 emin) fiyat ve teslimatı mükemmel teşekkürler
- (model %100 emin) ürünü beğendim gayet kaliteli

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
