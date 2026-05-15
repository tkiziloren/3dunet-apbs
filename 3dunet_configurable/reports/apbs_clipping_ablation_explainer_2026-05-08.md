# APBS Clipping and Ablation Explainer - 2026-05-08

Bu doküman APBS feature'larında kullandığımız clipping/normalization varyantlarını ve ablation mantığını savunmada anlatılabilecek açıklıkta özetler.

## 1. Clipping Nedir?

Clipping, bir feature değerini belirli bir alt ve üst sınır arasına sıkıştırmaktır.

Genel formül:

```text
x_clipped = min(max(x, min_value), max_value)
```

Örnek:

```text
range = [-20, +20]
x = +45  -> +20
x = -80  -> -20
x = +12  -> +12
```

APBS için clipping yapmamızın sebebi şudur:

- APBS electrostatic potential değerleri çok geniş aralıklara gidebilir.
- Çok yüksek pozitif/negatif uç değerler neural network training'i domine edebilir.
- Aşırı uç değerleri sınırlayarak modelin genel spatial pattern'i öğrenmesini kolaylaştırırız.
- Ancak fazla agresif clipping yaparsak gerçekten bilgi taşıyan yüksek potansiyel bölgeleri kaybedebiliriz.

Bu yüzden `clip5`, `clip10`, `clip20`, no-cutoff ve signed varyantları karşılaştırıyoruz.

## 2. Bizim Kodda Genel Normalization Akışı

Kodda APBS feature için ana fonksiyon:

```text
normalize_feature(feature_array, "electrostatic_grid", feature_normalization)
```

Genel akış:

```text
1. Raw APBS value okunur: x
2. Eğer clip=true ise:
   x = clip(x, min, max)
3. Eğer normalize=true ise:
   x = (x - min) / (max - min)
4. Eğer output_min/output_max verilmişse:
   x = x * (output_max - output_min) + output_min
5. Training transform içinde Standardize açıksa:
   x = (x - channel_mean) / channel_std
```

Önemli nokta:

`clip*_minmax` varyantları önce APBS'i `[0,1]` aralığına çevirir. Fakat training sırasında `Standardize(channel_wise=True)` uygulanıyorsa modelin gördüğü son tensor tekrar z-score yapılır. Yani modelin son gördüğü değerler tam olarak `[0,1]` kalmayabilir.

Bu clipping'i anlamsız yapmaz. Çünkü clipping, z-score öncesindeki dağılımı belirler ve uç değerlerin mean/std üzerinde aşırı etkili olmasını engeller.

## 3. APBS Clipping Varyantları

### 3.1 `apbs_clip5_minmax`

Tanım:

```text
x_clipped = clip(x, -5, +5)
x_out = (x_clipped + 5) / 10
```

Mapping:

```text
x <= -5  -> 0.0
x = 0    -> 0.5
x >= +5  -> 1.0
```

Yorum:

- En agresif clipping varyantıdır.
- Küçük electrostatic farkları korur.
- Fakat `-20`, `-50`, `-150` gibi değerlerin hepsi aynı `0.0` değerine düşer.
- `+20`, `+50`, `+150` gibi değerlerin hepsi aynı `1.0` değerine düşer.
- Eğer yüksek potansiyel bölgeleri binding pocket için anlamlıysa bu varyant bilgi kaybedebilir.

Savunma cümlesi:

> `clip5`, APBS sinyalinin sadece düşük voltaj aralığında yeterli olup olmadığını test eder. Kötü çıkarsa yüksek electrostatic magnitude bilgisinin önemli olabileceğini düşündürür.

### 3.2 `apbs_clip10_minmax`

Tanım:

```text
x_clipped = clip(x, -10, +10)
x_out = (x_clipped + 10) / 20
```

Mapping:

```text
x <= -10 -> 0.0
x = 0    -> 0.5
x >= +10 -> 1.0
```

Yorum:

- `clip5`'e göre daha geniş electrostatic aralık tutar.
- Hâlâ uç değerleri bastırır.
- Orta düzey potansiyel farklarının etkisini ölçmek için ara varyanttır.

Not:

`apbs_clip10` script içinde `apbs_clip10_minmax` alias'ıdır; aynı işlemi yapar.

### 3.3 `apbs_clip20_minmax`

Tanım:

```text
x_clipped = clip(x, -20, +20)
x_out = (x_clipped + 20) / 40
```

Mapping:

```text
x <= -20 -> 0.0
x = 0    -> 0.5
x >= +20 -> 1.0
```

Yorum:

- Şu ana kadarki deneylerde güçlü görünen varyanttır.
- `clip5` ve `clip10`'a göre daha yüksek electrostatic magnitude bilgisini korur.
- No-cutoff'a göre uç değerleri kontrol altında tutar.
- Bu yüzden iyi bir denge noktası olabilir.

Not:

`apbs_clip20` script içinde `apbs_clip20_minmax` alias'ıdır; aynı işlemi yapar.

Savunma cümlesi:

> `clip20`, APBS'in yüksek ama çok ekstrem olmayan electrostatic bölgelerini korurken training'i bozabilecek outlier değerleri sınırlar. Şu anki bulgular bu dengenin faydalı olabileceğini gösteriyor.

### 3.4 `apbs_no_cutoff_current`

Tanım:

```text
x_out = x
```

Kod ayarı:

```text
clip = false
normalize = false
```

Yorum:

- Raw APBS değerleri doğrudan dataset tensoruna girer.
- Eğer training transform olarak `Standardize(channel_wise=True)` açıksa, model raw değerlerin z-score edilmiş halini görür.
- Bu varyantta APBS'in uç değerleri korunur.
- Ama çok büyük pozitif/negatif değerler mean/std ve gradient davranışını etkileyebilir.

Savunma cümlesi:

> No-cutoff varyantı, APBS field'ındaki bütün magnitude bilgisini koruyunca model daha iyi mi öğreniyor yoksa outlier'lar training'i bozuyor mu sorusunu test eder.

Not:

`apbs_no_cutoff` script içinde `apbs_no_cutoff_current` alias'ıdır.

### 3.5 `apbs_full_minmax`

Tanım:

```text
x_out = (x + 150) / 300
```

Kod ayarı:

```text
min = -150
max = +150
clip = false
normalize = true
```

Yorum:

- Bu gerçek dataset min/max hesaplayıp normalize etmek değildir.
- Sabit fiziksel aralık varsayımıdır: `[-150, +150]`.
- `clip=false` olduğu için `x < -150` ise output `0` altına, `x > +150` ise output `1` üstüne çıkabilir.
- Eğer APBS değerlerinin çoğu gerçekten `[-150,+150]` içindeyse daha geniş ölçekli bir minmax representation sağlar.

Savunma cümlesi:

> `full_minmax`, APBS'in tüm geniş aralığını kırpmadan lineer olarak küçük sayısal aralığa taşımayı test eder. Burada amaç magnitude bilgisini kaybetmemek ama raw scale'i de modele doğrudan vermemektir.

### 3.6 `apbs_full_signed`

Tanım:

```text
x_tmp = (x + 150) / 300
x_out = x_tmp * 2 - 1
```

Basitleştirilmiş hali:

```text
x_out = x / 150
```

Kod ayarı:

```text
min = -150
max = +150
clip = false
normalize = true
output_min = -1
output_max = +1
```

Yorum:

- İşaret bilgisini doğal şekilde korur.
- `x = 0` değeri `0` olarak kalır.
- Pozitif APBS pozitif, negatif APBS negatif temsil edilir.
- `clip=false` olduğu için `|x| > 150` değerleri `[-1,+1]` dışına çıkabilir.

Savunma cümlesi:

> `full_signed`, APBS'in pozitif ve negatif işaretini model için daha açık hale getirir. Çünkü zero potential tam sıfırda kalır; bu, `[0,1]` mapping'e göre daha fiziksel okunabilir bir temsil olabilir.

### 3.7 `apbs_clip20_signed`

Tanım:

```text
x_clipped = clip(x, -20, +20)
x_tmp = (x_clipped + 20) / 40
x_out = x_tmp * 2 - 1
```

Basitleştirilmiş hali:

```text
x_out = clip(x, -20, +20) / 20
```

Mapping:

```text
x <= -20 -> -1.0
x = 0    ->  0.0
x >= +20 -> +1.0
```

Yorum:

- `clip20_minmax` ile aynı clipping sınırını kullanır.
- Fakat output `[0,1]` yerine `[-1,+1]` olur.
- Zero potential `0.5` değil, `0.0` olarak temsil edilir.
- APBS'in signed physical nature'ını daha doğal temsil eder.

Savunma cümlesi:

> `clip20_signed`, `clip20` bilgisini korur ama electrostatic sign'ı daha doğrudan temsil eder. Eğer bu varyant daha iyi çıkarsa, modelin pozitif/negatif potansiyel ayrımından faydalandığını düşünebiliriz.

### 3.8 `apbs_posneg_clip20`

Tanım:

Bu varyant APBS'i tek kanal yerine iki kanala böler:

```text
positive_channel = clip(x, 0, 20) / 20
negative_channel = clip(-x, 0, 20) / 20
```

Mapping:

```text
x = +20 -> positive=1.0, negative=0.0
x = +10 -> positive=0.5, negative=0.0
x = 0   -> positive=0.0, negative=0.0
x = -10 -> positive=0.0, negative=0.5
x = -20 -> positive=0.0, negative=1.0
```

Yorum:

- Pozitif ve negatif electrostatic potential alanlarını ayrı feature kanalları yapar.
- Modelin tek signed kanaldan işaret öğrenmesini beklemek yerine, pozitif/negatif magnitude ayrımını açıkça veririz.
- Bu bazen convolution modelleri için daha kolay öğrenilebilir.
- Dezavantajı channel sayısını artırmasıdır.

Savunma cümlesi:

> `posneg_clip20`, APBS alanını iki fiziksel moda ayırır: pozitif potansiyel bölgeleri ve negatif potansiyel bölgeleri. Bu, modelin sign bilgisini daha açık kullanmasını sağlayabilir.

## 4. Clipping Sonrası Standardize Detayı

Training config içinde `Standardize(channel_wise=True)` varsa, her feature kanalı için şu işlem yapılır:

```text
x_final = (x - channel_mean) / channel_std
```

Bu şu anlama gelir:

- `clip20_minmax` önce `[0,1]` üretir.
- Sonra channel-wise standardization bunu mean 0, std 1 olacak şekilde değiştirir.
- Dolayısıyla modelin son gördüğü değerler `[0,1]` olmak zorunda değildir.

Ama clipping hâlâ önemlidir:

- Çünkü standardization öncesi outlier'ları sınırlar.
- Mean/std'nin ekstrem değerler tarafından bozulmasını azaltır.
- Feature dağılımını daha stabil hale getirir.

Savunma cümlesi:

> Clipping ve standardization aynı şey değildir. Clipping hangi fiziksel APBS aralığını koruyacağımızı belirler; standardization ise training optimizasyonu için sayısal ölçeği düzenler.

## 5. Ablation Ne Demek?

`Ablation`, bir sistemi parçalara ayırıp her parçanın katkısını ölçmek demektir.

Türkçede birebir doğal karşılığı yok. "Parça çıkarma deneyi", "bileşen katkı analizi" veya "tek değişkenli kontrollü karşılaştırma" denebilir.

En net açıklama:

> Ablation, modelde veya inputta bir bileşeni çıkarıp/değiştirip, diğer her şeyi aynı tutarak bu bileşenin başarıya etkisini ölçmektir.

## 6. Ablation Neden Gerekli?

Eğer sadece en iyi sonucu veren modeli gösterirsek şu sorular cevapsız kalır:

- Başarı APBS'ten mi geliyor?
- Shape zaten yeterli miydi?
- Kimyasal feature'lar mı asıl katkıyı veriyor?
- Model mimarisi mi sonucu değiştirdi?
- Label tanımı mı daha kolaydı?
- Threshold mu skoru şişirdi?

Ablation bu soruları ayırır.

## 7. Bizim Çalışmadaki Ablation Türleri

### 7.1 Feature Ablation

Amaç:

```text
Hangi input feature başarıya ne kadar katkı veriyor?
```

Örnekler:

```text
shape_only
apbs_only
shape + apbs
shape + selected_chem
apbs + shape + selected_chem
```

Yorum:

- `shape_only` güçlü ise geometri çok bilgi taşıyor demektir.
- `apbs_only` güçlü ise electrostatic field bağımsız sinyal taşıyor demektir.
- `shape + apbs`, `shape_only`'den iyiyse APBS geometriye ek katkı sağlıyor demektir.
- `apbs + shape + selected_chem`, `shape + selected_chem`'den iyiyse APBS kimyasal feature'ların yanında da değerli demektir.

### 7.2 APBS Normalization Ablation

Amaç:

```text
APBS'i modele nasıl temsil etmek en iyi sonucu veriyor?
```

Örnekler:

```text
apbs_clip5_minmax
apbs_clip10_minmax
apbs_clip20_minmax
apbs_no_cutoff_current
apbs_full_minmax
apbs_full_signed
apbs_clip20_signed
apbs_posneg_clip20
```

Yorum:

- Bu deney APBS'in sadece var/yok etkisini değil, nasıl sunulması gerektiğini ölçer.
- Eğer `clip20` iyi çıkıyorsa, orta-yüksek electrostatic magnitude bilgisinin yararlı olduğunu ama uç outlier'ların kontrol edilmesi gerektiğini düşünebiliriz.
- Eğer signed veya pos/neg split iyi çıkarsa, APBS işaret bilgisinin model için kritik olduğunu düşünebiliriz.

### 7.3 Architecture Ablation

Amaç:

```text
Aynı feature ile hangi model ailesi APBS sinyalini daha iyi öğreniyor?
```

Örnekler:

```text
UNet3D4L
UNet3D4LA
ResNet3D4L
ResidualUNet3D
SEResUNet3D
CBAMUNet3D
UNetPlusPlus3D
TinyConvNeXtUNet3D
```

Yorum:

- Eğer APBS sadece bir modelde iyi çalışıyorsa sonuç daha zayıftır.
- Eğer farklı mimarilerde tutarlı katkı veriyorsa APBS sinyali daha güvenilir görünür.

### 7.4 Label Ablation

Amaç:

```text
Ground truth binding site tanımı sonucu ne kadar değiştiriyor?
```

Örnekler:

```text
binding_site_calculated
binding_site_in_dataset
binding_site_fpocket_selected
```

Yorum:

- Binding site için tek evrensel label yoktur.
- Ligand-distance label, dataset cavity label ve fpocket pocket label farklı şeyleri temsil edebilir.
- Bu yüzden label kaynağını değiştirmek de kontrollü bir deneydir.

### 7.5 Grid Resolution Ablation

Amaç:

```text
36, 72 ve 161 grid çözünürlüklerinde model/APBS davranışı nasıl değişiyor?
```

Yorum:

- 36 grid hızlıdır ve Kalasanty/PUResNet benzeri çalışmalarla daha yakın ölçek verir.
- 72 ara çözünürlük sağlar.
- 161 APBS'in native gridine daha yakın olabilir ama memory ve class imbalance maliyeti artar.
- Daha büyük grid otomatik olarak daha iyi sonuç anlamına gelmez.

### 7.6 Loss / Pos Weight Ablation

Amaç:

```text
Class imbalance ile nasıl başa çıkmak en iyi sonucu veriyor?
```

Örnekler:

```text
pos_weight = 10
pos_weight = 25
pos_weight = 50
pos_weight = 100
```

Yorum:

- Çok düşük `pos_weight`: model pocket tahmin etmekten kaçabilir.
- Çok yüksek `pos_weight`: model aşırı büyük pocket tahmin edebilir.
- Precision/recall dengesi bu yüzden değişir.

## 8. Ablation Ne Değildir?

Ablation şunlar değildir:

- Rastgele hyperparameter aramak.
- En iyi çıkan koşuyu seçip sadece onu göstermek.
- Aynı anda feature, model, label, threshold ve loss değiştirip sonucu yorumlamak.
- Test setine bakarak threshold veya model seçmek.

Doğru ablation için:

- Tek seferde mümkün olduğunca tek değişken değişmeli.
- Diğer koşullar sabit kalmalı.
- Negatif sonuçlar da raporlanmalı.
- Sonuç foldlar veya external dataset ile doğrulanmalı.

## 9. Bu Çalışma Sunulmaya Değer mi?

Objektif kanaatim: Evet, sunulmaya değer. Ama nasıl sunulduğu çok önemli.

Güçlü sunum şekli:

> Bu çalışma, APBS tabanlı electrostatic potential representation'ın 3D protein-ligand binding-site segmentation için bağımsız ve tamamlayıcı bir sinyal olup olmadığını sistematik ablation deneyleriyle test eder.

Zayıf sunum şekli:

> Mevcut modellere bir APBS kanalı ekledim ve bazen daha iyi oldu.

İkisi arasında büyük fark var.

## 10. Neden Sunulmaya Değer?

### 10.1 APBS Fiziksel Olarak Anlamlı

APBS rastgele üretilmiş bir kanal değildir. Protein çevresindeki electrostatic potential field'i temsil eder. Ligand binding'de electrostatic complementarity, polar interactions ve charge distribution önemlidir. Bu yüzden APBS'in binding site prediction için denenmesi biyofiziksel olarak mantıklıdır.

### 10.2 APBS-only Sonuçları Bilimsel Olarak İlginç

Eğer APBS-only anlamlı DCC/Pocket-F1/DVO üretiyorsa bu önemli bir bulgudur:

```text
Protein shape veya atomic one-hot olmadan, electrostatic field tek başına pocket localization sinyali taşıyor.
```

Bu sonuç negatif çıksa bile değerlidir; pozitif çıkarsa daha da güçlüdür.

### 10.3 APBS'in DVO'ya Katkısı Özellikle Önemli

DCC sadece pocket merkezini ölçer. DVO pocket hacminin/şeklinin örtüşmesini ölçer. APBS combined modellerde DVO'yu artırıyorsa, APBS sadece "yaklaşık doğru yeri" değil, pocket hacmini/şeklini de iyileştiriyor olabilir.

Bu makale için güçlü bir mesajdır.

### 10.4 Çalışma Sadece Model Değil, Representation Çalışması

Buradaki ana soru:

```text
Protein binding-site prediction'da electrostatic field nasıl temsil edilmeli?
```

Bu doktora seviyesinde savunulabilir bir sorudur. Çünkü feature representation, label definition, metric choice ve model architecture birlikte ele alınmaktadır.

## 11. Hangi Koşullarda Yayın İçin Güçlü Olur?

Bu çalışma hakemli makale için güçlü hale gelir, eğer:

1. APBS-only farklı foldlarda tutarlı anlamlı başarı gösterirse.
2. `shape + apbs`, `shape_only`'den tutarlı iyi çıkarsa.
3. `apbs + shape + selected_chem`, APBS'siz karşılığından iyi çıkarsa.
4. APBS özellikle DVO veya DCC/Pocket-F1 tarafında katkı verirse.
5. Sonuçlar sadece küçük local sette değil, full scPDB/PDBBind veya BU48/COACH üzerinde de görülürse.
6. Threshold ve postprocess validation'da seçilip testte sabit uygulanırsa.
7. Feature leakage içeren kanallar inputtan çıkarılmış olursa.
8. Negatif sonuçlar da raporlanırsa.

## 12. Şu Anki Zayıf Noktalar

Objektif olmak gerekirse şu an zayıf noktalar var:

- Local deneyler dataset büyüklüğü açısından sınırlı.
- Full benchmark sonuçları tamamlanmadan "state-of-the-art" iddiası kurulamaz.
- APBS preprocessing protonation/charge/grid alignment kararlarına duyarlı.
- 36/72/161 çözünürlüklerde fiziksel karşılaştırma dikkatli yapılmalı.
- Threshold sweep yanlış anlatılırsa "test set tuning" gibi algılanabilir.
- Sadece tek fold veya tek split üzerinden güçlü tez iddiası kurulmamalı.
- Kalasanty/PUResNet ile adil karşılaştırma için aynı benchmark ve metrik protokolü gerekir.

## 13. Benim Net Kanaatim

### Doktora savunması için

Evet, sunulmaya değer. Çünkü:

- Problem gerçek ve önemli.
- APBS fiziksel olarak anlamlı bir hipotez.
- Pipeline sadece engineering değil; label, metric, representation ve model etkilerini ölçüyor.
- Kendi hatalarını ve metrik sorunlarını fark edip düzelten bir araştırma süreci var.

### Hakemli makale için

Potansiyel var, ama final sonuçlara bağlı.

Makale mesajı şu olursa güçlü olur:

> APBS-derived electrostatic potential provides an independent and complementary signal for 3D protein-ligand binding-site segmentation, with measurable improvements especially in pocket localization and/or volume overlap under controlled ablation studies.

Makale zayıf kalır, eğer:

- APBS sadece bir küçük splitte iyi çıkarsa.
- Combined modellerde APBS katkısı tutarlı olmazsa.
- External benchmark yapılmazsa.
- Kalasanty/PUResNet karşılaştırması aynı metriklerle kurulmazsa.
- Sonuçlar sadece çok fazla deneme içinden seçilmiş en iyi run gibi görünürse.

## 14. Savunmada Kullanılacak Kısa Cevap

> Clipping, APBS electrostatic potential değerlerini belirli fiziksel aralıkta sınırlandırıp modele daha stabil bir dağılım vermek için kullanıldı. `clip5`, `clip10`, `clip20`, no-cutoff, signed ve pos/neg split varyantlarını denememizin sebebi, APBS bilgisinin hangi temsilinin binding-site segmentation için daha yararlı olduğunu ölçmekti. Bu bir ablation çalışmasıdır: tek bir bileşeni değiştirip diğerlerini sabit tutarak o bileşenin katkısını ölçüyoruz. Bence bu çalışma sunulmaya değer, çünkü sadece bir feature ekleme değil; fiziksel electrostatic field representation'ın bağımsız ve tamamlayıcı katkısını sistematik olarak test ediyor. Ancak güçlü yayın iddiası için foldlar, external benchmarklar ve adil Kalasanty/PUResNet metrik karşılaştırmaları tamamlanmalıdır.

