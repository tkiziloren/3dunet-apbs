# Thesis Defense Question Bank - 2026-05-08

Bu doküman doktora tez savunmasında gelebilecek teknik sorulara hazırlanmak için oluşturuldu. Cevaplar savunmada konuşulabilecek dildedir: önce kısa cevap, sonra komite derinleştirirse verilecek teknik açıklama.

## 1. 3D U-Net nasıl çalışır?

**Kısa cevap:**  
3D U-Net, proteini 3 boyutlu voxel grid olarak alır ve her voxel için "binding site mi değil mi" olasılığı üretir. Encoder kısmı lokal ve global yapısal örüntüleri öğrenir, decoder kısmı bu bilgiyi tekrar orijinal çözünürlüğe taşır. Skip connection'lar, kaybolabilecek ince uzamsal bilgiyi decoder'a geri verir.

**Detaylı cevap:**  
Bu problem bir 3D segmentation problemidir. Girdi, örneğin `36 x 36 x 36 x C` veya `161 x 161 x 161 x C` boyutunda çok kanallı bir grid olabilir. Kanallar shape, APBS electrostatic potential veya atomik/kimyasal feature'lar olabilir. Modelin çıktısı `D x H x W x 1` boyutunda bir logit/probability mask olur.  

Encoder tarafında 3D convolution blokları protein çevresindeki lokal 3D örüntüleri öğrenir. Pooling veya stride ile grid küçülür; böylece model daha geniş bağlamı görür. Decoder tarafında upsampling ile tekrar orijinal grid boyutuna dönülür. Skip connection sayesinde yüzey/cep sınırı gibi detaylar korunur. Bu yüzden U-Net, binding pocket gibi lokal ama bağlam gerektiren segmentation problemleri için doğal bir mimaridir.

## 2. Kullandığın modellerin farkını biliyor musun?

**Kısa cevap:**  
Evet. Hepsi aynı problemi çözüyor ama bilgiyi farklı biçimde işliyor. Basit U-Net daha doğrudan encoder-decoder yapısıdır. Attention U-Net skip connection'lardan gelen bilgiyi filtreler. ResNet tabanlı modeller residual bağlantılarla derinleşmeyi kolaylaştırır. SE/CBAM modeller kanal veya uzamsal dikkat ekler. UNet++ decoder skip bağlantılarını daha yoğun ve kademeli yapar. ConvNeXt tarzı modeller modern convolution blokları kullanır ama 3D'de daha ağır olabilir.

**Detaylı cevap:**  
Bizim model ailelerini şöyle anlatırım:

- `UNet3D4L`: Dört seviyeli klasik 3D U-Net. Baseline olarak önemli, çünkü az karmaşık ve yorumlaması kolay.
- `UNet3D4LA`: U-Net'e attention gate eklenmiş hali. Decoder'a aktarılan skip feature'ları otomatik olarak ağırlıklandırır; yani her düşük seviye detay eşit derecede taşınmaz.
- `UNet3D4LStrided`: Downsampling'i max-pooling yerine strided convolution ile yapar. Bu sayede küçültme işlemi de öğrenilebilir hale gelir.
- `ResNet3D4L`: Residual bağlantılar kullanır. Amaç, derin ağlarda gradient akışını iyileştirmek ve "identity mapping" üzerinden daha stabil öğrenmektir. Work5'te APBS-only için en güçlü tamamlanan model buydu.
- `ResidualUNet3D`: U-Net yapısına residual bloklar ekler. Segmentasyon yapısını korurken daha iyi feature extraction hedefler.
- `SEResUNet3D`: Squeeze-and-Excitation ekler. Kanalların önemini öğrenir; örneğin APBS kanalının bazı koşullarda daha baskın kullanılmasına izin verir.
- `CBAMUNet3D`: Kanal attention ve spatial attention kullanır. Hem "hangi kanal önemli?" hem de "gridin hangi bölgesi önemli?" sorularını öğrenmeye çalışır.
- `UNetPlusPlus3D`: U-Net skip bağlantılarını daha yoğun ve kademeli hale getirir. Encoder-decoder semantic gap'i azaltmayı hedefler.
- `ConvNeXtUNet3D` / `TinyConvNeXtUNet3D`: Modern convolution tasarım fikirlerini 3D segmentation'a taşır. Büyük versiyon pahalı ve yavaş olabilir; bu yüzden daha küçük versiyonlar daha mantıklı olabilir.

Savunmada önemli nokta: Bu modeller rastgele seçilmedi. Model farklarını APBS representation etkisinden ayırmak için önce feature ablation, sonra architecture ablation yapıyoruz.

## 3. Diğer çalışmalara sadece bir feature eklemişsin. Bu doktora tezi için yeterli mi?

**Kısa cevap:**  
Eğer çalışma sadece "mevcut modele bir kanal daha ekledim" seviyesinde kalırsa zayıf olur. Ama bu çalışmanın ana katkısı sadece bir kanal eklemek değil; APBS tabanlı elektrostatik potansiyelin binding-site segmentation için ne kadar bilgi taşıdığını sistematik olarak test etmek, farklı feature setleri, label tanımları, metrikler, grid çözünürlükleri ve mimariler altında bunu izole etmektir.

**Detaylı cevap:**  
Tezin katkısını şöyle kurmak daha doğru:

1. APBS electrostatic potential'ın tek başına binding-site localization sinyali taşıyıp taşımadığını test ediyoruz.
2. APBS'in shape ve kimyasal atomik feature'lara eklenince DCC, DCA, DVO ve Pocket-F1 üzerindeki etkisini ölçüyoruz.
3. APBS normalizasyonunun etkisini inceliyoruz: `clip5`, `clip10`, `clip20`, no-cutoff, signed normalization, positive/negative split.
4. Kalasanty/PUResNet benzeri paper-style metriklerle değerlendirme yapıyoruz.
5. Farklı mimarilerde APBS'in davranışını test ediyoruz.
6. Aynı pipeline ile scPDB, PDBBind, BU48 ve COACH gibi benchmarklara genişleme planlıyoruz.

Yani savunma cümlesi şu olabilir:  
"Benim iddiam sadece yeni bir kanal eklemek değil; elektrostatik potansiyelin 3D binding-site segmentation'da bağımsız ve tamamlayıcı bir sinyal olup olmadığını kontrollü deneylerle ölçmek."

## 4. Neden standart bir library kullanmadın da hepsini kendin geliştirdin?

**Kısa cevap:**  
Başlangıçta standart library daha hızlı olurdu, ama bu tezde feature generation, label tanımı, threshold, postprocess, metric ve cache alignment gibi ayrıntıları kontrol etmem gerekiyordu. Hazır bir library bu süreçleri çoğu zaman saklıyor. Bu yüzden configurable ve izlenebilir bir pipeline kurdum.

**Detaylı cevap:**  
Burada kritik olan şey sadece modeli eğitmek değil. Protein gridinin nasıl oluşturulduğu, APBS gridinin proteinle nasıl hizalandığı, label'ın liganddan mı cavity'den mi üretildiği, hangi feature'ların leakage taşıdığı ve metric'in nasıl hesaplandığı doğrudan sonucu değiştiriyor.  

Hazır library kullansaydım şu sorulara cevap vermem zorlaşırdı:

- APBS grid protein grid ile gerçekten aynı koordinat sisteminde mi?
- `binding_site_calculated` ile dataset'in verdiği cavity label aynı şeyi mi temsil ediyor?
- Model başarısı voxel-F1 ile mi, DCC ile mi, DVO ile mi ölçülüyor?
- Threshold değişince model başarısı neden değişiyor?
- Feature leakage var mı?
- Cache üretiminde hangi proteinler neden başarısız oldu?

Bu yüzden custom pipeline bilimsel olarak daha savunulabilir hale geldi. Buna rağmen model bileşenlerinde MONAI gibi standart kaynaklar ve literatürdeki U-Net/ResNet/attention fikirleri referans alınıyor.

## 5. Threshold sweep nedir? Image segmentation'da genelde sabit threshold kullanılır, neden burada sweep yapıyoruz?

**Kısa cevap:**  
Threshold sweep, modelin verdiği olasılık haritasını farklı eşiklerde binary mask'e çevirip performansın nasıl değiştiğini görmektir. Sabit threshold'u resmi karşılaştırma için tutuyoruz; sweep'i ise modelin kalibrasyonunu ve potansiyelini anlamak için kullanıyoruz. Test setinde en iyi threshold'u seçip tek başarı gibi sunmak doğru olmaz; ama validation üzerinde threshold seçip sonra testte sabitlemek bilimsel olarak kabul edilebilir.

**Detaylı cevap:**  
Model her voxel için bir skor üretir. Bu skor 0 ile 1 arasındadır ama "0.5 üzeri kesin pocket" demek her zaman doğru değildir. Özellikle bu problem çok sparse'tır: gridin çok küçük kısmı pocket, çok büyük kısmı background'dur. Class imbalance, loss ağırlıkları ve label boyutu nedeniyle modelin olasılıkları iyi kalibre olmayabilir.

Bu yüzden üç şeyi ayrı tutuyoruz:

1. **Fixed threshold score:** Örneğin threshold 0.40 veya 0.50. Bu resmi ve karşılaştırılabilir skor.
2. **Best validation threshold:** Validation üzerinde en iyi görünen threshold. Bu modelin hangi eşikte anlamlı çalıştığını gösterir.
3. **Test threshold:** Final değerlendirmede validation'dan seçilen threshold sabitlenmeli ve testte değiştirilmemelidir.

Savunmada net cümle:  
"Threshold sweep bir trick değildir; test setine bakarak threshold seçersem trick olur. Ben sweep'i diagnostik ve validation-based calibration için kullanıyorum."

## 6. Neden 0.50 threshold her zaman doğru değil?

**Kısa cevap:**  
0.50 doğal görünür ama model çıktısı kalibre olasılık değilse 0.50 biyolojik veya istatistiksel olarak özel bir eşik değildir. Sparse segmentation'da 0.50 bazen hiç voxel seçmeyebilir; daha düşük veya daha yüksek threshold ise daha doğru pocket localization verebilir.

**Detaylı cevap:**  
Binary classification'da 0.50 ancak şu koşullarda doğal kabul edilir:

- Pozitif/negatif sınıf maliyeti simetrikse,
- Model olasılıkları iyi kalibre edilmişse,
- Train ve test dağılımı uyumluysa,
- Pozitif sınıf çok aşırı sparse değilse.

Binding-site segmentation'da bu varsayımlar zayıftır. Bu yüzden fixed threshold'u raporlamak gerekir, ama threshold curve ve best validation threshold'u görmeden modelin gerçekten öğrenip öğrenmediğini anlamak eksik olur.

## 7. Diğer modeller 36 grid ile başarı elde ederken sen 161 ile ancak az geçiyorsan bu neden unique olsun?

**Kısa cevap:**  
Unique olan sadece daha büyük grid kullanmak değil. 161 APBS'in doğal grid çözünürlüğüne daha yakın olabilir ama yüksek çözünürlük tek başına başarı garantisi vermez. Özgünlük, elektrostatik potansiyel temsilinin hangi çözünürlükte, hangi normalizasyonda ve hangi model ailesiyle binding-site localization'a katkı verdiğini sistematik olarak göstermektir.

**Detaylı cevap:**  
36 grid hızlı ve literatürle karşılaştırmaya uygun olabilir. Kalasanty benzeri yaklaşımlar 70 Angstrom fiziksel alanı 2 Angstrom spacing ile temsil eder; bu yaklaşık 36 grid hücresi yapar. 161 grid ise APBS potansiyel alanını daha ayrıntılı taşıyabilir, fakat:

- Daha fazla voxel daha fazla memory ve daha zor class imbalance demektir.
- Model kapasitesi artmazsa yüksek çözünürlüğü kullanamayabilir.
- Label daha ince hale geldiğinde voxel-F1 daha katı olabilir.
- Küçük dataset yüksek çözünürlükte overfitting riskini artırabilir.

Bu yüzden başarıyı sadece "161 kullandım, daha iyi olmalı" diye beklememek gerekir. Savunmada güçlü pozisyon şu:  
"36, 72 ve 161 çözünürlükleri karşılaştırarak APBS bilgisinin çözünürlük-duyarlılığını ölçüyorum. Eğer 36'da bile APBS-only anlamlı sonuç verirse, bu APBS sinyalinin güçlü olduğunu gösterir. 161'de iyileşirse, native electrostatic field detayının ek katkısını gösterir."

## 8. Neden multimodal approach kullanmadın?

**Kısa cevap:**  
Aslında channel-level multimodal bir yaklaşım kullanıyoruz: shape, electrostatic potential ve kimyasal/atomik kanalları aynı 3D grid üzerinde birleştiriyoruz. Eğer multimodal'den sequence + graph + grid gibi farklı veri modalitelerini kastediyorsak, bu iyi bir gelecek çalışma ama bu tezin odağı APBS etkisini izole etmek.

**Detaylı cevap:**  
Multimodal yaklaşım iki anlama gelebilir:

1. Aynı grid içinde farklı fiziksel/kimyasal kanallar: Biz bunu yapıyoruz.
2. Farklı temsil aileleri: sequence model, atom graph, surface mesh, voxel grid gibi yapıların birlikte kullanılması. Bu daha karmaşık bir çalışma.

Bu tezde önce şu soruya cevap vermek gerekiyor:  
"APBS electrostatic potential tek başına veya shape/chemistry ile birlikte ne kadar katkı sağlıyor?"

Eğer en baştan sequence, graph, surface ve voxel hepsi birlikte kullanılsaydı APBS'in katkısını izole etmek zorlaşırdı. Bu yüzden kontrollü deney tasarımı açısından önce grid tabanlı ablation daha doğru.

## 9. Attention ve convolution blokları ne işe yarar?

**Kısa cevap:**  
Convolution blokları 3D komşuluk örüntülerini öğrenir. Attention blokları ise modelin hangi kanal veya hangi bölgeye daha fazla önem vereceğini öğrenmesine yardım eder. Binding pocket probleminde convolution cep geometrisini öğrenir; attention ise protein yüzeyindeki daha ilgili bölgeleri veya APBS/kimyasal kanalların önemini öne çıkarabilir.

**Detaylı cevap:**  
3D convolution bir voxel'in çevresindeki lokal hacmi tarar. İlk katmanlar kenar, yüzey, küçük boşluk gibi lokal özellikleri; daha derin katmanlar daha geniş pocket geometry ve fizikokimyasal örüntüleri öğrenebilir.  

Attention çeşitleri:

- **Skip attention:** Encoder'dan decoder'a giden düşük seviye detayları filtreler.
- **Channel attention / SE:** Hangi feature kanalının daha önemli olduğunu öğrenir.
- **CBAM:** Hem kanal hem spatial attention uygular.
- **Transformer attention:** Daha global ilişki kurabilir ama 3D gridde memory maliyeti çok yüksektir.

Savunma cümlesi:  
"Attention'ı sihirli bir başarı aracı olarak değil, APBS ve kimyasal kanalların nerede ve ne kadar kullanılacağını öğrenebilecek bir mekanizma olarak değerlendiriyorum."

## 10. Neden graph kullanmadın?

**Kısa cevap:**  
Graph protein atom bağlantılarını temsil etmek için güçlüdür, ama bizim çıktımız 3D pocket mask olduğu için voxel/grid tabanlı segmentation daha doğrudan bir seçimdir. Graph kullanmak mümkün ve değerli bir gelecek çalışma; fakat APBS grid alanı ve DVO gibi volumetrik metriklerle en doğal temsil 3D grid'dir.

**Detaylı cevap:**  
Graph neural network atomları node, bağları veya uzaysal yakınlıkları edge olarak temsil eder. Bu representation atom-level ilişkiler için güçlüdür. Ancak binding-site prediction'da çıktıyı genellikle 3D pocket volume veya pocket center olarak değerlendirmek istiyoruz. DCC, DCA ve DVO gibi metrikler de uzaysal hacim ile doğrudan ilişkilidir.

APBS zaten bir 3D scalar field üretir. Bunu graph'a çevirmek mümkün ama bilgi kaybı veya tasarım karmaşıklığı doğurabilir. Bu yüzden ilk aşamada grid mantıklıdır. Gelecek çalışma olarak graph-grid fusion önerilebilir:

- Atom graph protein kimyasını öğrenir.
- APBS grid elektrostatik alanı verir.
- Surface mesh yüzey geometrisini verir.
- Final head pocket segmentation veya pocket ranking yapar.

## 11. APBS neden bu iş için anlamlı bir feature?

**Kısa cevap:**  
Ligand binding sadece geometrik boşluk meselesi değildir; elektrostatik uyum, yük dağılımı ve polar etkileşimler de önemlidir. APBS, protein çevresindeki electrostatic potential alanını fiziksel modele dayalı olarak verir. Bu yüzden binding pocket localization için shape ve atomik feature'ları tamamlayabilecek bir sinyal taşıyabilir.

**Detaylı cevap:**  
Shape-only model boşlukları ve yüzey kıvrımlarını öğrenir. Atomik feature'lar lokal kimyasal bilgiyi verir. APBS ise daha uzun menzilli elektrostatik alanı temsil eder. İki pocket geometrik olarak benzer olabilir ama elektrostatik potansiyelleri farklı olabilir. Ligand bağlanmasında charged/polar bölgeler kritik olduğu için APBS'in DCC veya DVO üzerinde katkı vermesi biyofiziksel olarak anlamlıdır.

Savunmada temkinli cümle:  
"APBS tek başına tüm binding-site problemini çözmez; ama fiziksel olarak anlamlı ve shape/chemistry'den farklı bir alan bilgisi sağladığı için tamamlayıcı feature olarak değerlidir."

## 12. APBS feature'ında normalization neden bu kadar önemli?

**Kısa cevap:**  
APBS voltaj/potential değerleri çok geniş aralıklarda olabilir. Neural network bu değerleri doğrudan görürse çok yüksek uç değerler öğrenmeyi bozabilir veya model sadece ekstrem değerlere odaklanabilir. Bu yüzden clipping ve normalization APBS bilgisinin modele nasıl sunulduğunu ciddi şekilde değiştirir.

**Detaylı cevap:**  
APBS raw değerlerinde ekstrem pozitif/negatif potansiyeller olabilir. Bu değerleri:

- `clip5`: düşük aralığa sıkıştırırsak ekstrem bilgiyi kaybedebiliriz.
- `clip20`: daha geniş ama kontrollü bir aralık verir.
- no-cutoff: tüm ekstrem değerleri bırakır, ama training instabil olabilir.
- signed normalization: işaret bilgisini korur.
- positive/negative split: pozitif ve negatif alanları iki ayrı kanal olarak verir.

Work3/Work4'te `clip20` iyi göründüyse bu, yüksek potansiyel bölgelerinin bilgi taşıdığını ama tamamen kontrolsüz bırakmanın ideal olmadığını düşündürür.

## 13. `dist2ligand` gibi ligand-derived feature'lar neden riskli?

**Kısa cevap:**  
Prediction sırasında ligand genelde bilinmez. Eğer training'de modele ligand'a uzaklık gibi bilgi verirsek, model gelecekte erişemeyeceği bilgiyi öğrenmiş olur. Bu leakage'dir. Bu yüzden deploy edilecek modelde ligand-derived feature kullanılmamalıdır.

**Detaylı cevap:**  
Training, validation ve testte ligand varsa bile gerçek kullanım senaryosunda sadece protein verilecek. Modelden binding site tahmin etmesini bekleyeceğiz. Bu yüzden feature'lar prediction anında hesaplanabilir olmalı:

- Kullanılabilir: protein shape, protein atom types, APBS, hydrophobicity, protein surface features.
- Riskli/leakage: distance-to-ligand, ligand mask, ligand atom type, ligand center'a doğrudan bağlı feature.

Label üretmek için ligand kullanılabilir; çünkü ground truth binding site liganddan türetilebilir. Ama input feature olarak ligand bilgisi verilmemelidir.

## 14. Binding-site label'ı nasıl tanımlanıyor? Dataset label ile calculated label farkı nedir?

**Kısa cevap:**  
Calculated label genellikle ligand çevresindeki atomlardan veya mesafeden türetilir. Dataset label ise scPDB gibi kaynakların verdiği cavity/pocket tanımıdır. İkisi aynı şeyi temsil etmeyebilir. Bu yüzden ikisini ayrı ayrı test etmek bilimsel olarak önemlidir.

**Detaylı cevap:**  
Binding site için tek evrensel tanım yoktur. Bir çalışmada pocket ligand atomlarına yakın protein voxelleri olarak tanımlanabilir. Başka bir çalışmada cavity detection aracıyla bulunan boşluk hacmi kullanılabilir. scPDB tarafında cavity dosyaları, ligandın bulunduğu/ilişkili cavity'yi temsil eder.  

Bu fark sonucu etkiler:

- Ligand-distance label daha lokal olabilir.
- Cavity label daha hacimsel olabilir.
- Çok büyük cavity label DVO'yu etkileyebilir.
- Çok küçük label voxel-F1'i zorlaştırabilir.

Savunma cümlesi:  
"Label tanımı model başarısının parçasıdır. Bu yüzden APBS etkisini hem calculated label hem dataset-provided cavity label altında test etmek gerekir."

## 15. DCC, DCA, DVO, Pocket-F1 ve voxel-F1 arasındaki fark nedir?

**Kısa cevap:**  
Voxel-F1 mask seviyesinde voxel overlap ölçer. DCC predicted pocket center ile gerçek pocket center arasındaki mesafedir. DCA predicted pocket center ile ligand atomları arasındaki en yakın mesafeye bakar. DVO predicted ve actual volume overlap ölçer. Pocket-F1 ise pocket-level başarıyı özetler; voxel bazlı değil, pocket doğru bulundu mu sorusuna daha yakındır.

**Detaylı cevap:**  
Bu metrikleri şöyle ayırırım:

- **voxel-F1:** Her voxel'i binary classification gibi değerlendirir. Çok katı olabilir; iyi konumlanmış ama şekli farklı pocket düşük skor alabilir.
- **DCC:** Pocket merkezleri yakın mı? 4 Angstrom altı genellikle başarılı localization olarak kabul edilir.
- **DCA:** Predicted center ligand atomlarına yakın mı? Ligand merkezinden ziyade atomlara yakınlık açısından değerlidir.
- **DVO:** Predicted pocket hacmi gerçek pocket hacmiyle ne kadar örtüşüyor? Shape kalitesi için daha katıdır.
- **Pocket-F1:** Pocket-level detection başarısını özetler. Kalasanty/PUResNet tarzı karşılaştırmada daha anlamlı olabilir.

Savunmada önemli cümle:  
"Eski voxel-F1 skorlarımız modeli olduğundan başarısız gösterebilirdi; çünkü literatürde asıl vurgu pocket localization metrikleri olan DCC/DCA/DVO üzerindedir."

## 16. Modelin eski F1 skorları neden düşük çıkıyordu?

**Kısa cevap:**  
Çünkü eski ölçüm voxel-level F1'a daha fazla yaslanıyordu. Binding-site problemi sparse olduğu için voxel-F1 çok katı ve threshold'a duyarlıdır. Literatürde Kalasanty ve PUResNet gibi çalışmalar pocket localization metrikleriyle, özellikle DCC/DCA/DVO ve pocket-level başarıyla değerlendirilir.

**Detaylı cevap:**  
Voxel-F1 şu durumlarda düşük çıkabilir:

- Predicted pocket doğru yerde ama biraz büyükse FP artar.
- Predicted pocket doğru yerde ama label ile sınırları uyuşmuyorsa overlap düşük olur.
- Threshold çok yüksekse hiç voxel seçilmeyebilir.
- Threshold çok düşükse tüm grid pocket gibi seçilebilir.
- Label çok küçükse birkaç voxel kayması F1'i ciddi düşürür.

Bu yüzden voxel-F1'i tamamen atmıyoruz, ama model seçimi ve literatür karşılaştırması için paper-style metrikleri de raporluyoruz.

## 17. Selection score nedir ve neden kullanıyoruz?

**Kısa cevap:**  
Selection score, checkpoint seçerken tek bir metriğe aşırı bağlı kalmamak için oluşturulmuş birleşik bir değerlendirme skorudur. Ama final makale/tez sonucunda DCC, DCA, DVO, Pocket-F1 ve voxel-F1 ayrı ayrı raporlanmalıdır.

**Detaylı cevap:**  
Sadece DCC'ye göre checkpoint seçersek model doğru merkeze yakın ama aşırı büyük pocket üretebilir. Sadece DVO'ya göre seçersek şekil örtüşmesi iyi ama detection sayısı düşük olabilir. Sadece voxel-F1 seçersek literatürle karşılaştırma zayıflar. Bu yüzden checkpoint selection için birleşik skor pratik olabilir.  

Ama savunmada şunu net söylemek gerekir:  
"Selection score model seçmek için internal bir yardımcı metriktir; bilimsel karşılaştırmayı ham metriklerle yapıyorum."

## 18. Model overfitting yapıyor mu?

**Kısa cevap:**  
Overfitting'i tek bir log satırından söylemek doğru değil. Train metriği artarken validation düşüyorsa, validation DCC/DVO bozuluyorsa veya predicted pocket size uçuyorsa overfitting olabilir. Şu ana kadar bazı koşularda uzun epochlarda iyileşme gördük, ama bunu foldlar ve bağımsız benchmarklarla doğrulamak gerekiyor.

**Detaylı cevap:**  
Bu problemde overfitting sadece train loss ile anlaşılmaz. Bakılması gerekenler:

- Train voxel-F1 artarken validation pocket metrics düşüyor mu?
- Validation predicted pocket size epochlar arasında aşırı oynuyor mu?
- DCC artarken DVO düşüyor mu?
- Model validation'da sadece belli protein family'lerinde mi başarılı?
- Farklı foldlarda aynı feature set tutarlı mı?

Bu yüzden k-fold, external benchmark ve per-protein analiz kritik.

## 19. Neden k-fold yapıyoruz?

**Kısa cevap:**  
Tek split şanslı veya şanssız olabilir. K-fold, modelin farklı validation proteinlerinde tutarlı davranıp davranmadığını gösterir. Tez savunmasında tek run yerine fold ortalaması daha güçlü kanıttır.

**Detaylı cevap:**  
Protein datasetlerinde family similarity, pocket tipi ve ligand sınıfı dağılımı çok önemlidir. Tek validation split, kolay veya zor proteinleri fazla içerebilir. K-fold bu etkiyi azaltır.  

Finalde ideal raporlama:

- Her fold için DCC/DCA/DVO/Pocket-F1.
- Ortalama ve standart sapma.
- En iyi modelin sadece tek split'te değil, foldlar arasında tutarlı olduğunu gösterme.

## 20. APBS-only güçlü çıkarsa bunu nasıl yorumlarsın?

**Kısa cevap:**  
APBS-only'nin güçlü çıkması, elektrostatik potansiyel alanının pocket localization için bağımsız sinyal taşıdığını gösterir. Bu tez için önemli bir bulgudur, çünkü sadece atom type veya shape bilgisine dayanmadan fiziksel alan bilgisinin predictive olduğunu destekler.

**Detaylı cevap:**  
APBS-only başarısı şu anlama gelir:

- Protein yüzeyindeki elektrostatik pattern binding pocket ile ilişkili olabilir.
- APBS, lokal atomik feature'lardan farklı ve tamamlayıcı bilgi taşıyabilir.
- APBS normalization doğru yapılırsa model bu bilgiyi kullanabilir.

Ama şu temkin gerekli:

- APBS başarısı dataset bias'tan etkilenmiş olabilir.
- scPDB deep cavity ağırlıklı olabilir.
- External benchmark ile test edilmeden genelleme iddiası sınırlı kalır.

## 21. APBS'in DVO'ya katkısı neden önemli?

**Kısa cevap:**  
DCC sadece pocket merkezinin doğru olup olmadığına bakar. DVO ise predicted pocket hacminin gerçek pocket hacmiyle ne kadar örtüştüğünü ölçer. APBS DVO'yu artırıyorsa, sadece merkezi bulmakla kalmayıp pocket şekline de katkı sağlıyor olabilir.

**Detaylı cevap:**  
Bir model ligand çevresine yakın bir merkez bulabilir ama çok büyük veya yanlış şekilli bir pocket üretebilir. Bu durumda DCC iyi, DVO düşük olur. APBS'in DVO'yu iyileştirmesi elektrostatik alanın pocket sınırlarını veya pocket hacmini daha doğru belirlemeye yardım ettiğini düşündürür. Bu makale açısından güçlü bir mesaj olabilir.

## 22. Kalasanty ve PUResNet ile adil karşılaştırma nasıl yapılmalı?

**Kısa cevap:**  
Adil karşılaştırma için aynı veya benzer dataset, aynı train/test ayrımı, aynı metrikler ve benzer postprocess kullanılmalı. Kalasanty ve PUResNet'in kullandığı DCC/DCA/DVO/Pocket-F1 tarzı metriklerle raporlama yapmak bu yüzden önemli.

**Detaylı cevap:**  
Kalasanty 3D fully convolutional segmentation yaklaşımıdır ve scPDB üzerinde eğitilip DCC/DVO ile değerlendirilmiştir. PUResNet ve PUResNetV2.0 protein binding-site prediction için 3D/sparse representation ve benchmark datasetleri kullanır. Bu çalışmalarla karşılaştırmada dikkat edilmesi gerekenler:

- Dataset aynı mı?
- Holo/apo ayrımı var mı?
- Pocket label aynı şekilde mi çıkarıldı?
- Postprocess aynı mı?
- İlk pocket mi değerlendiriliyor, tüm pocketlar mı?
- DCC success threshold 4 Angstrom mu?
- DVO hangi mask üzerinden hesaplanıyor?

Savunma cümlesi:  
"Literatürle doğrudan yarış iddiası için aynı benchmark üzerinde aynı metrik ve postprocess ile raporlamam gerekir. Bu yüzden BU48/COACH ve scPDB full/fold deneylerini planlıyorum."

## 23. Kalasanty 36 grid ile başarılıysa bizim 36 gridde iyi sonuç almamız gerekir mi?

**Kısa cevap:**  
Gerekir demek fazla kesin olur. Kalasanty'nin input representation'ı, label üretimi, postprocess'i, dataset filtresi, training süresi ve mimarisi farklı. Ancak 36 grid bizim için önemli bir karşılaştırma noktasıdır; çünkü literatürde kabul görmüş pratik çözünürlüklerden biridir.

**Detaylı cevap:**  
36 grid şu avantajları verir:

- Daha hızlı training.
- Daha az memory.
- Literatürle daha yakın input scale.
- Daha düşük overfitting riski.

Ama dezavantajları:

- APBS'in ince alan bilgisi kaybolabilir.
- Küçük pocket detayları yumuşayabilir.
- Voxel spacing daha kaba olduğu için DVO etkilenebilir.

Bu yüzden 36, 72 ve 161 birlikte incelenmeli.

## 24. Holo ve apo ne demek, neden önemli?

**Kısa cevap:**  
Holo yapı ligand bağlı protein yapısıdır. Apo yapı ligand bağlı olmayan protein yapısıdır. Holo'da pocket ligand etkisiyle daha belirgin olabilir; apo'da pocket kapanmış veya şekil değiştirmiş olabilir. Bu yüzden apo üzerinde başarı daha zor ama gerçek uygulamaya daha yakındır.

**Detaylı cevap:**  
Binding-site prediction'da birçok dataset holo yapılardan oluşur. Bu, modelin daha kolay sinyal görmesine yol açabilir çünkü ligand bağlıyken pocket açık ve organize haldedir. Apo yapılarda conformational change, induced fit ve yüzey hareketleri nedeniyle pocket daha az belirgin olabilir.  

Savunma cümlesi:  
"Bu tezin ilk aşaması holo ağırlıklı benchmarklarda APBS sinyalini izole etmek. Daha güçlü genelleme iddiası için apo/holo ayrımı ayrıca test edilmelidir."

## 25. Neden postprocess gerekiyor?

**Kısa cevap:**  
Model voxel olasılık haritası üretir. Bunu biyolojik anlamlı pocketlara çevirmek için threshold, connected components, küçük komponentleri temizleme ve border temizleme gibi postprocess adımları gerekir. Bu adımlar Kalasanty/PUResNet tarzı pocket-level değerlendirmeye daha uygun sonuç verir.

**Detaylı cevap:**  
Raw mask bazen dağınık küçük adacıklar veya grid kenarında anlamsız prediction üretebilir. Postprocess şunları yapabilir:

- Threshold ile binary mask üretir.
- Morphological closing ile küçük boşlukları kapatır.
- Border-touching komponentleri temizler.
- Connected components ile pocket adaylarını ayırır.
- Minimum volume filtresiyle çok küçük noise parçalarını atar.
- En büyük veya en yüksek skorlu komponenti pocket olarak seçer.

Ancak postprocess de train edilmeyen bir karar katmanıdır. Bu yüzden parametreleri validation üzerinde belirleyip testte sabitlemek gerekir.

## 26. Model çıktılarını gerçek prediction'da nasıl kullanacaksın?

**Kısa cevap:**  
Gerçek prediction'da sadece protein verilecek. Protein üzerinden shape, atomik feature'lar ve APBS gibi ligand gerektirmeyen feature'lar hesaplanacak. Model olasılık haritası üretecek. Daha sonra sabit threshold ve postprocess ile pocket adayları çıkarılacak.

**Detaylı cevap:**  
Serve senaryosu:

1. Kullanıcı protein yapısını verir.
2. Protein standardize edilir.
3. Grid oluşturulur.
4. APBS hesaplanır veya önceden hesaplanmış grid alınır.
5. Seçili feature kanalları oluşturulur.
6. Model probability map üretir.
7. Validation'dan belirlenmiş threshold uygulanır.
8. Connected components ile pocket adayları çıkarılır.
9. Pocket center, volume ve confidence score raporlanır.

Ligand-derived input kullanılmaz.

## 27. Bu çalışma neden publication çıkarabilir?

**Kısa cevap:**  
Publication potansiyeli var, özellikle APBS'in DVO ve pocket localization metriklerine tutarlı katkısı gösterilebilirse. Ancak makale için tek run yetmez; k-fold, external benchmark, APBS ablation, normalization analizi ve literatürle aynı metriklerde karşılaştırma gerekir.

**Detaylı cevap:**  
Yayın mesajı şu olabilir:

"Electrostatic potential fields computed by APBS provide an independent and complementary signal for 3D protein-ligand binding-site segmentation."

Bunu desteklemek için gerekli kanıtlar:

- APBS-only anlamlı başarı.
- APBS + shape > shape-only.
- APBS + selected chemistry > selected chemistry veya shape+chemistry.
- DVO katkısı özellikle raporlanmalı.
- Foldlar arasında tutarlılık.
- BU48/COACH/PDBBind gibi external benchmark.
- Normalization sensitivity analizi.

## 28. Tezin en zayıf noktası ne olabilir?

**Kısa cevap:**  
En zayıf nokta, sonuçların dataset ve label tanımına bağımlı olması olabilir. Ayrıca APBS hesaplaması protonation, charge assignment, grid alignment ve normalization kararlarına duyarlıdır.

**Detaylı cevap:**  
Komite şu eleştirileri getirebilir:

- scPDB deep cavity ağırlıklı, gerçek dünyayı tam temsil etmiyor.
- Dataset-provided cavity ile ligand-derived label farklı.
- APBS hesaplaması pH/protonation ayarına duyarlı.
- 36/72/161 karşılaştırması fiziksel voxel spacing açısından dikkatli kurulmalı.
- Threshold/postprocess test setine göre optimize edilirse sonuç şişebilir.
- Protein family leakage varsa model genelleme yerine benzer proteinleri öğrenmiş olabilir.

Bunlara cevap:

- K-fold ve external benchmark planlanıyor.
- Label kaynakları ayrı raporlanıyor.
- APBS preprocessing açıkça dokümante ediliyor.
- Threshold validation üzerinde seçilip testte sabitleniyor.
- Feature leakage içeren kanallar eğitim kombinasyonlarına dahil edilmiyor.

## 29. APBS hesaplama/protein hazırlama hataları sonucu etkiler mi?

**Kısa cevap:**  
Evet, ciddi şekilde etkiler. APBS grid proteinle hizalı değilse, charge/protonation hatalıysa veya scaling yanlışsa model yanlış sinyal öğrenir. Bu yüzden cache validation ve grid alignment kontrolü bu tezin kritik parçasıdır.

**Detaylı cevap:**  
APBS feature için kontrol edilmesi gerekenler:

- APBS grid origin protein grid origin ile uyumlu mu?
- Voxel spacing ve target span doğru mu?
- 161 APBS grid 36/72 gridlere indirgenirken fiziksel koordinatlar korunuyor mu?
- Charge assignment tutarlı mı?
- Protein hazırlama pipeline'ı train/test boyunca aynı mı?
- Extreme potential değerleri nasıl normalize ediliyor?

Bu kontroller yapılmadan APBS'in iyi/kötü olduğu söylenemez.

## 30. Neden sadece highest score pocket değil, DCC/DCA/DVO ayrı ayrı raporlanmalı?

**Kısa cevap:**  
Çünkü tek skor modelin hatasını saklayabilir. DCC doğru merkezi, DCA liganda yakınlığı, DVO hacimsel örtüşmeyi, voxel-F1 mask kalitesini gösterir. Hepsi birlikte modelin neyi iyi neyi kötü yaptığını anlatır.

**Detaylı cevap:**  
Bir model:

- DCC iyi, DVO kötü olabilir: doğru yere yakın ama çok büyük pocket.
- DVO iyi, DCC kötü olabilir: küçük overlap var ama merkezi uzak.
- Voxel-F1 düşük, DCC iyi olabilir: localization doğru ama mask sınırı farklı.
- Pocket-F1 iyi, precision düşük olabilir: birçok cep tahmin edip gerçek cebi de yakalıyor olabilir.

Bu yüzden tek metrikle başarı iddiası kurmak zayıftır.

## 31. Class imbalance sorununu nasıl ele alıyorsun?

**Kısa cevap:**  
Binding-site voxel'ları gridin çok küçük kısmını oluşturduğu için class imbalance çok güçlü. Loss ağırlığı, threshold analizi, paper-style metrics ve per-protein raporlama ile bunu kontrol ediyoruz.

**Detaylı cevap:**  
Positive voxel çok az olduğunda model tüm grid'i background tahmin ederek düşük loss alabilir. Tam tersi, pos_weight çok yüksekse model çok fazla voxel'i pocket tahmin edebilir. Bu yüzden:

- `pos_weight` ablation yapılmalı.
- Dice/Focal/BCE kombinasyonları denenebilir.
- Fixed threshold ve best validation threshold birlikte raporlanmalı.
- Predicted pocket size her epoch takip edilmeli.
- Precision/recall ayrı görülmeli.

## 32. Prediction çok büyük pocket üretiyorsa bu ne anlama gelir?

**Kısa cevap:**  
Model recall'u artırıp precision'u düşürüyor olabilir. Yani gerçek pocket'ı yakalıyor ama yanında çok fazla yanlış voxel de seçiyor olabilir. Bu durumda DCC iyi görünebilir ama DVO ve voxel-F1 düşer.

**Detaylı cevap:**  
Bu problemi anlamak için şunlara bakılır:

- Average predicted pocket size.
- FP voxel sayısı.
- DCC success ama DVO düşük mü?
- Threshold yükselince pocket küçülüyor mu?
- Connected components sonrası ana component makul mü?

## 33. APBS yüksek voltajları kırpmak bilgi kaybı yaratır mı?

**Kısa cevap:**  
Evet, yaratabilir. Bu yüzden clip değerlerini deneysel olarak karşılaştırıyoruz. İlk bulgular `clip20` aralığının `clip5` veya no-cutoff'a göre daha iyi bir denge sağlayabileceğini düşündürüyor.

**Detaylı cevap:**  
Çok agresif clipping yüksek elektrostatik bölgeleri aynı değere sıkıştırır. No-cutoff ise ekstrem değerleri modele fazla baskın gösterebilir. Signed veya positive/negative split yaklaşımları işaret bilgisini daha anlamlı taşıyabilir. Work7'nin amacı bunu sistematik test etmek.

## 34. Neden shape-only bazen çok güçlü çıkıyor?

**Kısa cevap:**  
Çünkü birçok ligand binding pocket geometrik olarak cavity/cleft yapısındadır. Shape, özellikle scPDB gibi deep cavity ağırlıklı datasetlerde güçlü bir sinyal taşıyabilir.

**Detaylı cevap:**  
Bu APBS'in gereksiz olduğu anlamına gelmez. Shape zaten güçlü bir baseline olabilir. APBS'in değeri şurada ortaya çıkar:

- Shape-only'nin kaçırdığı charged/polar pocketlarda katkı veriyor mu?
- DVO'yu iyileştiriyor mu?
- Aynı DCC seviyesinde pocket volume daha doğru mu?
- External benchmarkta genelleme sağlıyor mu?

## 35. Bu kadar farklı deney yaparken multiple comparison / cherry-picking riski var mı?

**Kısa cevap:**  
Var. Bu yüzden deneyleri work package olarak önceden tanımlayıp, tüm sonuçları raporlamak ve final modeli validation protokolüyle seçmek gerekiyor. Sadece iyi çıkan run'ları göstermek bilimsel olarak zayıf olur.

**Detaylı cevap:**  
Savunmada şunu söylemek güven verir:

"Ablation çalışmalarını hipotez bazlı grupladım: feature ablation, APBS normalization, architecture sweep, grid resolution ve external benchmark. Negatif sonuçları da raporluyorum. Final iddia tek bir koşuya değil, tutarlı trendlere dayanacak."

## 36. APBS feature daha önce çalışılmadı mı? Novelty iddiasını nasıl kuracaksın?

**Kısa cevap:**  
"Hiç kimse APBS kullanmadı" gibi kesin bir iddia riskli olur. Daha güçlü ve savunulabilir iddia şu: APBS electrostatic potential'ın 3D binding-site segmentation'da bağımsız ve tamamlayıcı sinyal olarak sistematik ablation, normalization ve architecture analiziyle değerlendirilmesi.

**Detaylı cevap:**  
Novelty şu bileşenlerden gelir:

- APBS-only performansını ölçmek.
- APBS + shape/chemistry katkısını izole etmek.
- APBS normalization stratejilerini karşılaştırmak.
- DCC/DCA/DVO/Pocket-F1 ve voxel-F1 ile çok yönlü değerlendirmek.
- scPDB/PDBBind/BU48/COACH benchmarklarına genişletmek.
- APBS'in özellikle DVO üzerindeki etkisini incelemek.

Bu, "sadece feature ekleme" değil, fiziksel alan bilgisinin temsil ve genelleme gücünü analiz etme çalışmasıdır.

## 37. Neden PUResNetV2.0 gibi sparse representation kullanmadın?

**Kısa cevap:**  
Sparse representation büyük gridlerde memory açısından avantajlıdır. Bizim ilk hedefimiz APBS sinyalini kontrollü grid segmentation ortamında izole etmekti. Sparse model iyi bir gelecek adım; özellikle 161 grid ve daha büyük datasetlerde faydalı olabilir.

**Detaylı cevap:**  
Dense grid şunları kolaylaştırır:

- APBS scalar field ile doğal uyum.
- U-Net segmentation mimarileriyle doğrudan kullanım.
- DVO gibi volumetrik metriklerle kolay karşılaştırma.
- Cache/debug sürecinin daha izlenebilir olması.

Sparse representation ise:

- Büyük gridde daha verimli olabilir.
- Sadece protein çevresi veya yüzey voxellerine odaklanabilir.
- PUResNetV2.0 benzeri başarı için önemli olabilir.

Bu yüzden tezde dense APBS analizini tamamlayıp, makale/future work tarafında sparse APBS representation güçlü bir devam çalışmasıdır.

## 38. Model neden bazen geç öğreniyor?

**Kısa cevap:**  
APBS gibi continuous field feature'larında modelin anlamlı karar sınırı öğrenmesi daha uzun sürebilir. Ayrıca sparse label, loss ağırlığı, threshold ve normalization nedeniyle erken epochlar yanıltıcı olabilir.

**Detaylı cevap:**  
Geç öğrenmenin olası sebepleri:

- Pozitif voxel sayısı çok az.
- APBS değerlerinin ölçeği zor.
- Başlangıçta logits threshold üstüne çıkmıyor.
- Model önce background'u öğreniyor.
- Pocket localization metriği ancak belli representation oluşunca artıyor.

Bu yüzden 20 epoch smoke test sadece pipeline kontrolüdür; bilimsel sonuç için 100-250 epoch ve fold analizi gerekir.

## 39. Savunmada en güçlü ana mesaj ne olmalı?

**Kısa cevap:**  
"Bu tez, APBS tabanlı elektrostatik potansiyel alanının 3D protein-ligand binding-site segmentation'da bağımsız ve tamamlayıcı bir sinyal taşıyıp taşımadığını sistematik olarak test eder."

**Detaylı cevap:**  
Ana hikaye:

1. Problem: Binding-site prediction yalnızca geometriyle açıklanamaz; fizikokimyasal alanlar önemlidir.
2. Hipotez: APBS electrostatic potential binding-site localization ve pocket shape için ek sinyal sağlar.
3. Yöntem: 3D grid segmentation, APBS/shape/chemistry feature ablation, farklı label kaynakları, farklı metrikler.
4. Değerlendirme: DCC, DCA, DVO, Pocket-F1, voxel-F1, foldlar, external benchmarklar.
5. Bulgular: APBS-only anlamlı sinyal taşırsa ve combined modellerde DVO/Pocket-F1 artarsa hipotez desteklenir.
6. Katkı: APBS representation ve normalization'ın binding-site deep learning pipeline'ındaki etkisi sistematik olarak gösterilir.

## 40. Komite "bu engineering çalışması, doktora katkısı nerede?" derse ne denmeli?

**Kısa cevap:**  
Bu çalışma sadece software engineering değil; biyofiziksel bir representation'ın deep learning tabanlı binding-site prediction üzerindeki etkisini hipotez odaklı test ediyor. Engineering kısmı, hipotezi doğru ve tekrarlanabilir şekilde test etmek için gerekli altyapıdır.

**Detaylı cevap:**  
Doktora katkısı şu noktalarda:

- Fiziksel electrostatic field representation'ın predictive değerini ölçmek.
- Binding-site label ve metric tanımlarının sonucu nasıl değiştirdiğini göstermek.
- APBS normalization ve resolution etkisini analiz etmek.
- Feature leakage risklerini ayırmak.
- Literatür metrikleriyle adil karşılaştırma yapmak.
- Reproducible configurable pipeline sağlamak.

Savunma cümlesi:  
"Yazılım altyapısı katkının kendisi değil; katkıyı ölçülebilir ve savunulabilir yapan araçtır."

## 41. Komite "neden bu kadar çok model denedin?" derse ne denmeli?

**Kısa cevap:**  
Çünkü APBS'in etkisini tek bir mimariye bağlamak istemedim. Eğer APBS sadece bir modelde işe yarıyorsa bu zayıf kanıttır; farklı model ailelerinde tutarlı katkı veriyorsa daha güçlü bilimsel sonuçtur.

**Detaylı cevap:**  
Model sweep'in amacı leaderboard yapmak değil. Amaç şu soruları cevaplamak:

- APBS sinyali basit U-Net ile öğrenilebilir mi?
- Residual bağlantılar APBS öğrenimini kolaylaştırıyor mu?
- Attention APBS/chemistry kanallarını daha iyi seçiyor mu?
- ConvNeXt tarzı modern bloklar bu sparse 3D problemde avantajlı mı?

## 42. Komite "neden daha büyük dataset kullanmadın?" derse ne denmeli?

**Kısa cevap:**  
Yerel deneyleri küçük datasetle hızlı hipotez testi için yaptım. Final iddia için full scPDB/PDBBind ve external benchmarklar gerekir; bu yüzden cache generation ve SLURM planı hazırlandı.

**Detaylı cevap:**  
Küçük dataset:

- Pipeline debug için uygundur.
- Feature ablation hızlı yapılır.
- Modelin öğrenip öğrenmediği görülür.

Ama final sonuç için:

- Full scPDB foldları,
- PDBBind testleri,
- BU48/COACH external benchmarkları,
- Family-level split,
- Ortalama ve standart sapma gerekir.

## 43. Komite "APBS hesaplamak pahalı, pratik mi?" derse ne denmeli?

**Kısa cevap:**  
APBS ek maliyet getirir, ama binding-site prediction drug discovery pipeline'ında genellikle offline veya preprocessing adımıdır. Eğer APBS DVO/DCC üzerinde anlamlı katkı verirse bu maliyet makul olabilir. Ayrıca cache ve parallelization ile maliyet yönetilebilir.

**Detaylı cevap:**  
APBS'in maliyeti:

- Protein başına preprocessing süresi.
- Charge/protonation hazırlığı.
- Grid üretim maliyeti.

Ama avantajları:

- Fiziksel olarak yorumlanabilir feature.
- Model training'den bağımsız hesaplanabilir.
- Cache'lenebilir.
- Inference öncesi batch preprocessing yapılabilir.

## 44. Komite "APBS yanlış charge/protonation ile yanıltıcı olmaz mı?" derse ne denmeli?

**Kısa cevap:**  
Evet, bu önemli bir sınırlama. Bu yüzden protein preparation ve APBS parametreleri açıkça dokümante edilmeli. Bu çalışma APBS'in ideal mutlak fiziksel doğruluğundan çok, standardize edilmiş APBS representation'ın model için taşıdığı sinyali ölçer.

**Detaylı cevap:**  
APBS sonuçları şunlara bağlıdır:

- Protonation state.
- pH varsayımı.
- Force field charge assignment.
- Dielectric constants.
- Ionic strength.
- Grid spacing.

Bu sınırlama açıkça yazılmalı. İleri çalışma olarak farklı protonation/charge ayarlarına robustness analizi yapılabilir.

## 45. Komite "label zaten liganddan geliyor, model ligandı mı öğreniyor?" derse ne denmeli?

**Kısa cevap:**  
Label'ın liganddan türetilmesi normaldir; supervised learning'de ground truth böyle oluşturulur. Kritik olan ligand bilgisinin input feature olarak verilmemesidir. Prediction'da sadece protein tabanlı feature'lar kullanılacak.

**Detaylı cevap:**  
Label ve input ayrımı:

- Label: Eğitimde hedef olarak kullanılabilir; ligand konumu ground truth pocket tanımına yardım eder.
- Input: Prediction sırasında hesaplanabilecek protein-only feature'lardan oluşmalı.

Bu ayrımı net yapmak leakage eleştirisini azaltır.

## 46. Komite "DVO düşük ama DCC iyi, başarılı mı sayıyorsun?" derse ne denmeli?

**Kısa cevap:**  
Kullanım amacına bağlı. Docking grid merkezi bulmak için DCC iyi olabilir; pocket shape ve volume doğru olsun istiyorsak DVO önemlidir. Ben ikisini ayrı raporluyorum, çünkü biri diğerinin yerine geçmez.

**Detaylı cevap:**  
DCC docking için ilk arama bölgesini bulmada yeterli olabilir. Ama pocket segmentation kalitesi için DVO daha katıdır. Bu yüzden APBS'in DVO'ya katkısı özellikle önemli olabilir.

## 47. Komite "final modelini nasıl seçeceksin?" derse ne denmeli?

**Kısa cevap:**  
Final modeli validation protokolüne göre seçeceğim. Primary hedef DCC/Pocket-F1 ve DVO dengesi olacak. Threshold validation'dan belirlenecek, testte sabitlenecek. Ayrıca best selection, best Pocket-F1/DCC, best DVO, best fixed voxel-F1 ve final checkpoint ayrı saklanacak.

**Detaylı cevap:**  
Tek checkpoint yeterli olmayabilir. Bu yüzden şu kategoriler saklanmalı:

1. Best selection score.
2. Best Pocket-F1 / DCC.
3. Best DVO among DCC-successful predictions.
4. Best fixed-threshold voxel-F1.
5. Final epoch model.

Makale için primary checkpoint tanımı önceden belirtilmeli.

## 48. Komite "bu sonuçlar klinik veya drug discovery açısından ne ifade eder?" derse ne denmeli?

**Kısa cevap:**  
Bu çalışma doğrudan klinik karar aracı değildir. Erken aşama structure-based drug discovery'de docking veya pocket inspection için aday binding bölgelerini önermek üzere kullanılabilir. APBS katkısı, biyofiziksel olarak daha anlamlı pocket önerileri üretmeye yardımcı olabilir.

**Detaylı cevap:**  
Pratik kullanım:

- Protein üzerinde olası pocket merkezlerini bulmak.
- Docking search box önermek.
- Pocket ranking yapmak.
- Druggable/non-druggable analizine input sağlamak.

Ama ligand affinity, selectivity veya clinical efficacy tahmini değildir.

## 49. Komite "neden sadece protein structure, sequence bilgisi yok?" derse ne denmeli?

**Kısa cevap:**  
Bu tez structure-based binding-site prediction'a odaklanıyor. Sequence bilgisi evrimsel conservation sağlayabilir ama APBS ve pocket geometry etkisini izole etmek için ilk aşamada 3D structure representation kullandım.

**Detaylı cevap:**  
Sequence/evolutionary features eklenebilir:

- Conservation score.
- Multiple sequence alignment.
- Protein language model embeddings.

Ama bunlar APBS etkisini karıştırabilir. Gelecek çalışma olarak APBS + structure + sequence fusion güçlü bir yön olabilir.

## 50. Komite "senin çalışman Kalasanty/PUResNet'ten nasıl farklı?" derse ne denmeli?

**Kısa cevap:**  
Kalasanty ve PUResNet binding-site prediction için güçlü 3D deep learning yaklaşımlarıdır. Benim farkım, APBS electrostatic potential'ı ana hipotez olarak ele almam ve bu fiziksel alan bilgisinin tek başına ve diğer feature'larla birlikte katkısını sistematik ablationlarla ölçmemdir.

**Detaylı cevap:**  
Farklar:

- Kalasanty: 3D segmentation, scPDB, DCC/DVO, cavity detection odaklı.
- PUResNet: Protein binding-site prediction için 3D/sparse representation ve benchmarklar.
- Bu çalışma: APBS electrostatic field representation, normalization sensitivity, APBS-only vs APBS+shape/chemistry, DVO katkısı, configurable cache/metric pipeline.

Savunma cümlesi:  
"Ben Kalasanty/PUResNet'in yerine tamamen yeni bir problem tanımı koymuyorum; onların çizgisindeki 3D binding-site prediction problemine fiziksel electrostatic field representation'ın katkısını ölçüyorum."

## Ek Sorular - Kısa Hazırlık Listesi

Aşağıdaki sorular da savunmada gelebilir:

1. Protein family leakage'i nasıl önlüyorsun?
2. Train/validation/test split protein benzerliğine göre mi yapıldı?
3. APBS grid ile protein grid alignment'ını nasıl doğruladın?
4. Hangi feature'lar prediction sırasında hesaplanabilir?
5. `binding_site_calculated` ve `binding_site_in_dataset` sonuçları ayrışırsa hangisine güveneceksin?
6. Kötü çalışan proteinleri manuel inceledin mi?
7. Predicted pocket size dağılımı literatürle uyumlu mu?
8. DCC@4A neden kullanılıyor?
9. DCA ile DCC çelişirse nasıl yorumlarsın?
10. Voxel spacing değişince DVO nasıl etkilenir?
11. Threshold validation'da seçildiyse testte nasıl uygulanacak?
12. Postprocess parametrelerini nasıl sabitleyeceksin?
13. APBS hesaplanamayan proteinleri dışlamak bias yaratır mı?
14. Full dataset başarısı küçük dataset başarısından neden farklı olabilir?
15. Model başarısız olduğunda sebep feature mı, label mı, model mi nasıl ayırırsın?
16. Neden DCC/DCA/DVO yanında voxel-F1'i tamamen bırakmıyorsun?
17. APBS-only sonuç güçlü çıkarsa chemistry feature'lara gerek var mı?
18. APBS combined modelde katkı vermezse bunu nasıl yorumlarsın?
19. Runtime ve memory maliyeti nedir?
20. Bu model hangi protein tiplerinde çalışmaz?

## Kaynak Notları

- Kalasanty: Stepniewska-Dziubinska et al., "Improving detection of protein-ligand binding sites with 3D segmentation", Scientific Reports, 2020. https://www.nature.com/articles/s41598-020-61860-z
- PUResNetV2.0: Kandel et al., "PUResNetV2.0: a deep learning model leveraging sparse representation for improved ligand binding site prediction", Journal of Cheminformatics, 2024. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00865-6
- Apo/holo and Kalasanty usage context: Clark et al., "Predicting binding sites from unbound versus bound protein structures", Scientific Reports, 2020. https://www.nature.com/articles/s41598-020-72906-7

