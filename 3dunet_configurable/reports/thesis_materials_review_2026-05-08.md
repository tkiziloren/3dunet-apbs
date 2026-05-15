# Thesis Materials Review - 2026-05-08

Bu not, `/Users/tevfik/Sandbox/Tevfik/Projects/phd_examples/thesis_materials` klasörüne koyulan tez dosyalarının ilk incelemesidir.

## 1. Klasördeki Dosyalar

```text
Tevfik Kiziloren - Tez Onerisi Raporu.docx
TezYazımKılavuz_WORD_Şablon_12.12.2025.docx
TEZ-YAZIM-KILAVUZU_Eylül2022.pdf
OU-09(Tez_Sekilsel_Kontrolu_Basvurusu)_08042025.doc
ESOGU_FBE_TEZ_SEKILSEL_KONTROL_LISTESI(OGRENCI_ICIN)_06.01.2026.pdf
```

Okunabilirlik durumu:

- Tez önerisi `.docx`: okunabildi.
- Word tez şablonu `.docx`: okunabildi.
- Tez yazım kılavuzu PDF: `pypdf` ile metin çıkarılabildi.
- Şekilsel kontrol listesi PDF: `pypdf` ile metin çıkarılabildi.
- OU-09 `.doc`: `textutil` ile okunabildi.

## 2. Tez Önerisi Raporundan Çıkan Ana Hat

Mevcut tez önerisi başlığı:

```text
Protein Ligand Bağlanma Bölgelerinin Makine Öğrenimi Yöntemleriyle Tahmini
Protein Ligand Binding Site Prediction with Machine Learning Methods
```

Öneri raporunda ana fikir:

- Protein-ligand binding site prediction problemi anlatılmış.
- PDBbind, PDB, protein-ligand ilişkisi, APBS ve 3D U-Net açıklanmış.
- APBS electrostatic profile temel özgün feature olarak konumlandırılmış.
- 161 x 161 x 161 APBS grid, shape grid ve binding site mask gibi feature/label yapısı önerilmiş.
- Başlangıçta başarı metriği olarak F1 / MeanIoU gibi voxel-level segmentation ölçümleri düşünülmüş.

Güncel tez için değiştirilmesi gereken önemli noktalar:

- Eski öneride "herhangi bir kimyasal molekülün proteine bağlanıp bağlanmayacağı" gibi ifadeler var. Güncel çalışma ligand binding affinity veya ligand-specific binding prediction değil, protein üzerinde binding site / pocket localization problemidir.
- Eski öneride PDBbind ana dataset gibi duruyor. Güncel tezde scPDB, PDBbind, BU48/COACH gibi dataset ayrımları daha net kurulmalı.
- Eski öneride başarı ağırlığı F1 / MeanIoU gibi voxel metriklerde. Güncel tezde Kalasanty/PUResNet benzeri DCC, DCA, DVO, Pocket-F1 ve voxel-F1 birlikte anlatılmalı.
- Eski öneride "elektrostatik özelliklerin ilk defa kullanılması" iddiası güçlü ve riskli. Güncel tezde daha savunulabilir novelty cümlesi kullanılmalı:

```text
APBS-derived electrostatic potential representation is systematically evaluated as an independent and complementary signal for 3D protein-ligand binding-site segmentation.
```

## 3. Word Şablonundan Çıkan Tez Yapısı

Şablonda beklenen ön sayfalar:

- Türkçe dış kapak
- İngilizce dış kapak
- İç kapak
- Onay
- Etik Beyan
- Üretken Yapay Zeka Kullanımı Beyan Formu
- Özet
- Summary
- Teşekkür
- İçindekiler
- Şekiller Dizini
- Çizelgeler Dizini
- Simgeler ve Kısaltmalar Dizini
- Ana metin
- Kaynaklar Dizini
- Ekler

Önemli not:

Şablonda açıkça Üretken Yapay Zeka Kullanımı Beyan Formu var. Bu yüzden tez yazımında AI kullanımı saklanacak bir şey gibi değil, kurumun istediği şekilde beyan edilecek bir süreç olarak ele alınmalı.

## 4. Tez Yazım Kılavuzundan Kritik Format Kuralları

### Sayfa ve Yazı

- Kağıt: A4.
- Yazı tipi: Times New Roman.
- Tez metni: 12 punto.
- İçindekiler istenirse 10 veya 11 punto olabilir.
- Sayfa kenarları:
  - üst: 3 cm
  - sol: 3 cm
  - sağ: 2,5 cm
  - alt: 2,5 cm
- Metin iki yana yaslı olmalı.
- Standart satır aralığı: 1,5.
- Şekil/çizelge açıklamaları, dipnotlar ve kaynaklar: 1 tam aralık.
- Paragraflar: 1 tab / 1,25 cm içeriden başlamalı.

### Başlıklar

- Birinci derece bölüm başlıkları tamamen büyük harf, ortalanmış ve yeni sayfadan başlamalı.
- İkinci derece başlıklarda her kelimenin ilk harfi büyük olmalı ve başlık sola yaslı olmalı.
- Üçüncü ve dördüncü derece başlıklarda sadece ilk kelimenin baş harfi büyük, diğer kelimeler küçük olmalı.
- Dördüncü derece başlıklar altı çizili olmalı.
- Dördüncü dereceden daha ileri başlık kullanılmamalı.
- Tüm başlıklar koyu olmalı.

### Sayfa Numaraları

- Giriş öncesi sayfalar küçük Romen rakamıyla numaralanmalı: `i, ii, iii, ...`.
- Giriş ve Amaç ile başlayan ana metin Arap rakamlarıyla numaralanmalı: `1, 2, 3, ...`.
- Sayfa numarası sağ üstte olmalı.
- Dış kapak, iç kapak, onay, etik beyan gibi bazı sayfalarda numara sayılır ama görünmez.

### Şekil ve Çizelgeler

- Fotoğraf, grafik, histogram, harita gibi tüm görseller "Şekil" olarak adlandırılır.
- Tüm şekil ve çizelgelere metin içinde atıf yapılmalı.
- Şekil açıklaması şeklin altında olmalı.
- Çizelge açıklaması çizelgenin üstünde olmalı.
- Şekil ve çizelgeler bölüm bazlı numaralanmalı:

```text
Şekil 3.1
Şekil 3.2
Çizelge 4.1
Çizelge 4.2
```

### Kaynaklar

- Kılavuz APA 7.0 kaynak kurallarını veriyor.
- Tez içinde kullanılan her kaynağa metinde atıf yapılmalı.
- Kaynakçada olup metinde atıf yapılmayan kaynak bırakılmamalı.
- İnternet kaynaklarında erişim linki ve erişim tarihi isteniyor.

### Zorunlu Ana Bölümler

Kontrol listesine göre tez metninde şu ana bölümler bekleniyor:

- GİRİŞ veya GİRİŞ VE AMAÇ
- LİTERATÜR ARAŞTIRMASI
- TEORİK BİLGİ veya bunu karşılayan ana bölümler
- MATERYAL VE YÖNTEM
- BULGULAR VE TARTIŞMA
- SONUÇ VE ÖNERİLER

Bu tez için bu yapıya uymak mantıklı.

## 5. OU-09 Formundan Çıkan Süreç Notu

OU-09 formu tez savunması öncesi:

- tez şekilsel kontrolü,
- tez elektronik PDF kopyası,
- orijinallik raporu,
- danışmana rapor gönderimi

süreçleri için kullanılıyor.

Not:

Formda orijinallik raporunun "depo yok" seçeneğiyle alınacağı yazıyor.

## 6. Komite Profili İçin Tez Dilini Nasıl Ayarlamalıyız?

Kullanıcının belirttiği jüri profili:

```text
Elektrik-elektronik mühendisliği,
pattern recognition,
image processing.
```

Bu nedenle tez ve savunma anlatımında protein biyolojisi değil, sinyal/görüntü işleme benzetmesi güçlü tutulmalı.

Önerilen anlatım eşlemesi:

| Protein-domain kavramı | Jüriye anlatılacak pattern recognition karşılığı |
|---|---|
| Protein structure | 3D volumetric signal |
| Binding site | sparse 3D segmentation target |
| Voxel grid | 3D image / volumetric image |
| APBS electrostatic potential | physics-informed auxiliary channel |
| Shape channel | binary/occupancy geometry channel |
| Atomic features | semantic/chemical channels |
| 3D U-Net | volumetric encoder-decoder segmentation model |
| DCC/DCA | object localization distance metric |
| DVO | volumetric overlap metric |
| voxel-F1 | pixel/voxel-level segmentation F1 |
| threshold sweep | validation-based decision threshold calibration |
| postprocess | connected-component based object extraction |

Savunmada fazla biyolojiye girilmemeli. Protein background bölümü teze yeterince konur, ama sunumda ana hikaye şöyle kurulmalı:

```text
Bu çalışma 3D sparse segmentation problemidir. APBS, modelin inputuna fizik tabanlı bir ek kanal olarak eklenmiştir. Amaç, bu kanalın localization ve volumetric overlap performansına katkısını kontrollü ablation deneyleriyle ölçmektir.
```

## 7. Tez İçin Önerilen Güncel İçindekiler

100-150 sayfa hedefi için önerilen yapı:

```text
ÖZET
SUMMARY
TEŞEKKÜR
İÇİNDEKİLER
ŞEKİLLER DİZİNİ
ÇİZELGELER DİZİNİ
SİMGELER VE KISALTMALAR DİZİNİ

1. GİRİŞ VE AMAÇ
   1.1 Problem Tanımı
   1.2 Tezin Amacı
   1.3 Tezin Katkıları
   1.4 Tezin Organizasyonu

2. LİTERATÜR ARAŞTIRMASI
   2.1 Binding Site Prediction Yaklaşımları
   2.2 Geometry-based ve Energy-based Yaklaşımlar
   2.3 Deep Learning Tabanlı Yaklaşımlar
   2.4 3D CNN ve 3D U-Net Yaklaşımları
   2.5 Kalasanty
   2.6 PUResNet ve PUResNetV2.0
   2.7 Literatürdeki Boşluk ve Bu Tezin Konumu

3. TEORİK BİLGİ
   3.1 3D Görüntü ve Hacimsel Bölütleme
   3.2 Voxel Tabanlı Temsil
   3.3 Convolutional Neural Networks
   3.4 U-Net ve 3D U-Net
   3.5 Residual, Attention ve Modern Convolution Blokları
   3.6 Sparse Segmentation ve Class Imbalance
   3.7 Protein-Ligand Binding Site Probleminin Görüntü İşleme Yorumu
   3.8 APBS ve Electrostatic Potential Representation

4. MATERYAL VE YÖNTEM
   4.1 Datasetler
   4.2 Protein Hazırlama ve Cache Üretimi
   4.3 Grid Tanımı ve Koordinat Sistemi
   4.4 Label Üretimi
   4.5 Feature Kanalları
   4.6 APBS Clipping ve Normalization
   4.7 Model Mimarileri
   4.8 Training Protokolü
   4.9 Threshold ve Postprocess
   4.10 Metrikler
   4.11 Deney Tasarımı ve Ablation Planı

5. BULGULAR VE TARTIŞMA
   5.1 Baseline Deneyler
   5.2 Feature Ablation Sonuçları
   5.3 APBS-only Sonuçları
   5.4 APBS Normalization Sonuçları
   5.5 Model Architecture Sonuçları
   5.6 Combined Feature Sonuçları
   5.7 Fold Bazlı Değerlendirme
   5.8 Kalasanty/PUResNet ile Karşılaştırma
   5.9 Hata Analizi
   5.10 Tartışma

6. SONUÇ VE ÖNERİLER
   6.1 Genel Sonuçlar
   6.2 Tezin Bilimsel Katkısı
   6.3 Sınırlamalar
   6.4 Gelecek Çalışmalar

KAYNAKLAR DİZİNİ
EKLER
```

## 8. Eski Öneriden Yeni Teze Taşınabilecek Kısımlar

Doğrudan kullanılabilir / revize edilerek kullanılabilir:

- Protein ve ligand tanımı.
- PDB / PDBbind tanıtımı.
- Binding site kavramı.
- APBS ve Poisson-Boltzmann temeli.
- U-Net ve 3D U-Net tanımı.
- İlaç keşfi motivasyonu.

Dikkatli revize edilmeli:

- "İlk kez" novelty iddiası.
- "Herhangi bir molekül proteine bağlanır mı?" gibi ligand-specific ifadeler.
- Sadece PDBbind üzerinden anlatım.
- Başarım metriğinin sadece F1/MeanIoU gibi verilmesi.
- Pytorch-3dunet library kullanılacak denmiş; güncel tez configurable custom pipeline olarak anlatılmalı.

## 9. İlk Yazılabilecek Bölümler

Sonuçlar tamamen bitmeden yazılabilecek bölümler:

1. GİRİŞ VE AMAÇ
2. LİTERATÜR ARAŞTIRMASI
3. TEORİK BİLGİ
4. MATERYAL VE YÖNTEM'in pipeline, dataset, feature, model, metric kısımları

Sonuçlar bitmeden finalleştirilmemesi gereken bölümler:

1. BULGULAR VE TARTIŞMA
2. SONUÇ VE ÖNERİLER
3. ÖZET / SUMMARY

## 10. Çalışma Şekli Önerisi

En verimli yol:

1. Önce bu dosyalara göre tez iskeletini kilitle.
2. İngilizce teknik taslakları bölüm bölüm üret.
3. Kullanıcı Türkçeye kendi üslubuyla çevirsin.
4. Türkçe metin tekrar teknik doğruluk ve format açısından kontrol edilsin.
5. Sonuçlar geldikçe Bulgular ve Tartışma bölümüne tablolar ve yorumlar eklensin.
6. En son şablon `.docx` içine yerleştirilsin.

AI/orijinallik açısından öneri:

- Metin üretimi saklanmamalı; şablonda zaten Üretken Yapay Zeka Kullanımı Beyan Formu var.
- Kullanıcı nihai Türkçe metni kendi cümleleriyle yazmalı.
- Tüm deney sonuçları gerçek log/CSV/report dosyalarından gelmeli.
- Kaynaklar doğru verilmeli.
- Abartılı novelty iddiası kullanılmamalı.

