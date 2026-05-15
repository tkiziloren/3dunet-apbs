# Work8 Sonuç Raporu

Tarih: 2026-05-15

Çalışma klasörü:

`/Users/tevfik/Sandbox/github/PHD/runs/work8_combined_model_feature_representation_sweep_fold1_250epoch_thr040`

Work8, APBS içeren birleşik öznitelik setlerinde model mimarisi, öznitelik grubu ve APBS temsil biçiminin birlikte etkisini ölçmek için tasarlandı.

## Durum

Planlanan matris:

```text
5 model x 2 öznitelik grubu x 3 APBS temsil biçimi = 30 eğitim
```

Son durum:

```text
Tamamlanan eğitim: 30/30
Eksik eğitim: 0
Çalışan eğitim: 0
```

Kullanılan modeller:

- `UNetPlusPlus3D`
- `CBAMUNet3D`
- `UNet3D4LA`
- `ResNet3D4L`
- `ResNet3D4LGN`

Kullanılan öznitelik grupları:

- `apbs_shape`
- `apbs_shape_selected_chem`

Kullanılan APBS temsil biçimleri:

- `apbs_clip20_minmax`
- `apbs_full_signed`
- `apbs_posneg_clip20`

## Kısa Sonuç

Work8'in ana sonucu net:

**En güçlü model ailesi `UNetPlusPlus3D` oldu.**

En iyi genel koşu:

```text
UNetPlusPlus3D + apbs_shape + apbs_full_signed
```

Bu koşu hem seçim skoru hem cep düzeyi F1 hem de DCC açısından en iyi sonucu verdi.

```text
selection score: 1.8376
Pocket-F1:      0.7143
DCC@4A:         0.5556
DCA@4A:         0.7315
DVO(success):   0.5354
DVO(all):       0.4089
best epoch:     179
best threshold: 0.50
```

En iyi DVO(success) koşusu ise:

```text
UNetPlusPlus3D + apbs_shape_selected_chem + apbs_full_signed
```

```text
selection score: 1.8319
Pocket-F1:      0.6988
DCC@4A:         0.5370
DCA@4A:         0.7593
DVO(success):   0.5513
best epoch:     250
best threshold: 0.40
```

En iyi DCA koşusu:

```text
UNetPlusPlus3D + apbs_shape_selected_chem + apbs_posneg_clip20
```

```text
DCA@4A:         0.7685
Pocket-F1:      0.6909
DCC@4A:         0.5278
DVO(success):   0.5387
best epoch:     204
best threshold: 0.20
```

## En İyi 10 Koşu

| Sıra | Model | Öznitelik | APBS temsil | Selection | Pocket-F1 | DCC@4A | DCA@4A | DVO(success) | Epoch | Threshold |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | UNetPlusPlus3D | apbs_shape | full_signed | 1.8376 | 0.7143 | 0.5556 | 0.7315 | 0.5354 | 179 | 0.50 |
| 2 | UNetPlusPlus3D | apbs_shape_selected_chem | full_signed | 1.8319 | 0.6988 | 0.5370 | 0.7593 | 0.5513 | 250 | 0.40 |
| 3 | UNetPlusPlus3D | apbs_shape_selected_chem | posneg_clip20 | 1.8124 | 0.6909 | 0.5278 | 0.7685 | 0.5387 | 204 | 0.20 |
| 4 | UNetPlusPlus3D | apbs_shape | clip20_minmax | 1.8051 | 0.6909 | 0.5278 | 0.7500 | 0.5325 | 203 | 0.45 |
| 5 | CBAMUNet3D | apbs_shape | full_signed | 1.7999 | 0.7143 | 0.5556 | 0.7315 | 0.4917 | 144 | 0.40 |
| 6 | UNetPlusPlus3D | apbs_shape_selected_chem | clip20_minmax | 1.7998 | 0.6748 | 0.5093 | 0.7593 | 0.5351 | 207 | 0.55 |
| 7 | UNet3D4LA | apbs_shape | full_signed | 1.7974 | 0.6988 | 0.5370 | 0.7500 | 0.4994 | 218 | 0.45 |
| 8 | UNet3D4LA | apbs_shape_selected_chem | posneg_clip20 | 1.7868 | 0.6667 | 0.5000 | 0.7593 | 0.5305 | 151 | 0.55 |
| 9 | CBAMUNet3D | apbs_shape | posneg_clip20 | 1.7791 | 0.6909 | 0.5278 | 0.7130 | 0.5215 | 152 | 0.30 |
| 10 | UNet3D4LA | apbs_shape | clip20_minmax | 1.7684 | 0.6909 | 0.5278 | 0.7222 | 0.5178 | 216 | 0.30 |

## Model Ailesi Ortalamaları

| Model | Eğitim sayısı | Ortalama selection | Ortalama Pocket-F1 | Ortalama DCC@4A | Ortalama DCA@4A | Ortalama DVO(success) | En iyi selection |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNetPlusPlus3D | 6 | 1.8073 | 0.6894 | 0.5262 | 0.7485 | 0.5371 | 1.8376 |
| CBAMUNet3D | 6 | 1.7485 | 0.6827 | 0.5185 | 0.7145 | 0.5000 | 1.7999 |
| UNet3D4LA | 6 | 1.7444 | 0.6731 | 0.5077 | 0.7299 | 0.5096 | 1.7974 |
| ResNet3D4L | 6 | 1.7301 | 0.6730 | 0.5077 | 0.7145 | 0.5162 | 1.7678 |
| ResNet3D4LGN | 6 | 1.6794 | 0.6526 | 0.4846 | 0.6975 | 0.4866 | 1.7182 |

Yorum:

`UNetPlusPlus3D` yalnızca tek bir koşuda değil, altı koşunun ortalamasında da en iyi model ailesi oldu. Bu yüzden Work8 sonrası ana aday olarak alınmalı.

`ResNet3D4LGN` beklenenden zayıf kaldı. GroupNorm stabilite sağlayabilir; ancak bu matris içinde başarıyı artırmadı. Ana model adayı yapılmamalı.

## Öznitelik Grubu Ortalamaları

| Öznitelik grubu | Eğitim sayısı | Ortalama selection | Ortalama Pocket-F1 | Ortalama DCC@4A | Ortalama DCA@4A | Ortalama DVO(success) | En iyi selection |
|---|---:|---:|---:|---:|---:|---:|---:|
| apbs_shape_selected_chem | 15 | 1.7439 | 0.6697 | 0.5037 | 0.7296 | 0.5140 | 1.8319 |
| apbs_shape | 15 | 1.7399 | 0.6786 | 0.5142 | 0.7123 | 0.5058 | 1.8376 |

Yorum:

`apbs_shape` ortalamada daha iyi Pocket-F1 ve DCC verdi. Yani cep merkezini doğru yere koyma açısından daha temiz görünüyor.

`apbs_shape_selected_chem` ise DCA ve DVO(success) tarafında daha güçlü. Bu, kimyasal özniteliklerin cep hacmi ve ligand çevresiyle ilgili bilgiyi artırabileceğini gösteriyor.

Tezde bu ayrım şöyle yazılabilir:

> `apbs_shape` kombinasyonu cep lokalizasyonu açısından daha güçlü sonuç verirken, `apbs_shape_selected_chem` kombinasyonu ligand çevresi ve hacimsel örtüşme ölçütlerinde daha avantajlı davranmıştır.

## APBS Temsil Biçimi Ortalamaları

| APBS temsil biçimi | Eğitim sayısı | Ortalama selection | Ortalama Pocket-F1 | Ortalama DCC@4A | Ortalama DCA@4A | Ortalama DVO(success) | En iyi selection |
|---|---:|---:|---:|---:|---:|---:|---:|
| apbs_full_signed | 10 | 1.7666 | 0.6881 | 0.5250 | 0.7296 | 0.5093 | 1.8376 |
| apbs_posneg_clip20 | 10 | 1.7304 | 0.6706 | 0.5046 | 0.7176 | 0.5073 | 1.8124 |
| apbs_clip20_minmax | 10 | 1.7288 | 0.6637 | 0.4972 | 0.7157 | 0.5131 | 1.8051 |

Yorum:

`apbs_full_signed` açık ara en iyi genel temsil oldu. Bu önemli bir bulgu: APBS voltaj alanının işaret ve büyüklük bilgisini korumak, kırpılmış/minmax temsile göre daha iyi çalışıyor.

`apbs_clip20_minmax` ortalama DVO(success) tarafında küçük bir avantaj gösteriyor; ancak genel selection, Pocket-F1 ve DCC açısından `full_signed` daha güçlü.

`apbs_posneg_clip20` bazı koşullarda DCA açısından iyi sonuç verdi. Özellikle en iyi DCA sonucu bu temsil ile geldi.

## Metrik Bazında En İyi Koşular

| Metrik | En iyi koşu | Değer |
|---|---|---:|
| Selection score | UNetPlusPlus3D + apbs_shape + full_signed | 1.8376 |
| Pocket-F1 | UNetPlusPlus3D + apbs_shape + full_signed | 0.7143 |
| DCC@4A | UNetPlusPlus3D + apbs_shape + full_signed | 0.5556 |
| DCA@4A | UNetPlusPlus3D + apbs_shape_selected_chem + posneg_clip20 | 0.7685 |
| DVO(success) | UNetPlusPlus3D + apbs_shape_selected_chem + full_signed | 0.5513 |
| DVO(all) | UNetPlusPlus3D + apbs_shape + full_signed | 0.4089 |
| voxel-F1 | UNetPlusPlus3D + apbs_shape_selected_chem + clip20_minmax | 0.5473 |
| fixed threshold voxel-F1 | UNetPlusPlus3D + apbs_shape_selected_chem + clip20_minmax | 0.5457 |

## Epoch Yorumu

En iyi koşuların önemli bir bölümü geç epochlarda geldi:

- `UNetPlusPlus3D + apbs_shape + full_signed`: epoch 179
- `UNetPlusPlus3D + apbs_shape_selected_chem + full_signed`: epoch 250
- `UNetPlusPlus3D + apbs_shape_selected_chem + posneg_clip20`: epoch 204
- `UNetPlusPlus3D + apbs_shape + clip20_minmax`: epoch 203
- `UNet3D4LA + apbs_shape + full_signed`: epoch 218
- `ResNet3D4L + apbs_shape + full_signed`: epoch 206

Bu, 150 epochun bazı modeller için erken kalabileceğini gösteriyor. Özellikle `UNetPlusPlus3D + apbs_shape_selected_chem + full_signed` koşusunda en iyi sonuç epoch 250'de geldiği için bu koşu daha uzun eğitimden fayda görebilir.

Ancak her modeli daha uzun koşturmak verimli değil. Uzatma yapılacaksa yalnızca en iyi 2-3 aday üzerinde yapılmalı.

## Tez İçin Ana Bulgular

1. `UNetPlusPlus3D`, Work8 içinde en güçlü model ailesidir.
2. `apbs_full_signed`, genel başarı açısından en iyi APBS temsil biçimidir.
3. `apbs_shape`, DCC ve Pocket-F1 için en iyi ana adaydır.
4. `apbs_shape_selected_chem`, DCA ve DVO(success) tarafında avantaj sağlar.
5. Kimyasal öznitelikler lokalizasyonu her zaman artırmaz; fakat ligand çevresine yakınlık ve hacimsel örtüşme için faydalı olabilir.
6. ResNet tabanlı modeller hızlı ve makul sonuç vermesine rağmen Work8 içinde UNet++ seviyesine çıkamamıştır.
7. Eşik değerleri çoğunlukla 0.20-0.55 aralığında değişmiştir; en iyi genel koşunun threshold değeri 0.50'dir.
8. APBS voltaj alanının işaretli ve tam temsil edilmesi, kırpılmış temsile göre daha güçlü sinyal vermiştir.

## Kalasanty/PUResNet Eksenindeki Yorum

Bu sonuçlar Kalasanty/PUResNet çizgisinde anlamlıdır çünkü:

- Problem hâlâ 3B ızgara tabanlı bağlanma bölgesi tahminidir.
- Ana metrikler cep düzeyi F1, DCC, DCA ve DVO'dur.
- En iyi sonuçlar yalnızca voksel-F1 üzerinden değil, cep lokalizasyonu ve hacimsel örtüşme üzerinden de değerlendirilmektedir.

Work8'in en iyi koşusu:

```text
Pocket-F1: 0.7143
DCC@4A:    0.5556
DCA@4A:    0.7315
DVO:       0.5354
```

Bu değerler scPDB fold1 validation üzerinde elde edilmiştir. Bu yüzden literatürle kesin karşılaştırma için BU48, COACH420, PDBbind ve tam katlamalı scPDB sonuçları ayrıca gereklidir.

## Work8 Sonrası A/B/C Planı

Work8 ana sweep tamamlandı. Bundan sonra Work8'i büyütmek yerine üç küçük devam paketiyle kapatmak daha doğru.

### Work8A: Top-k Metrik Genişletmesi

Amaç:

Mevcut tamamlanmış Work8 modelleri üzerinde yeniden eğitim yapmadan daha ayrıntılı literatür uyumlu metrikler çıkarmak.

Yapılacaklar:

- Top-1, Top-2, Top-3 ve mümkünse Top-(n+2) cep değerlendirmesi eklenecek.
- Her cep bileşeni için DCC, DCA, DVO, hacim, merkez koordinatı ve skor yazılacak.
- Eski CSV dosyaları bozulmayacak.
- Yeni çıktılar ayrı CSV dosyalarına yazılacak:
  - `validation_paper_metrics_topk.csv`
  - `validation_paper_metrics_per_protein_topk.csv`
  - `work8_topk_reevaluation_summary.csv`

Neden önemli:

Kalasanty, PUResNet ve P2Rank/SwinSite çizgisinde top-k protokol farkları olabilir. Bizim başarımızı adil göstermek için top-1 ve top-3 başarıları ayrı raporlamamız gerekiyor.

Öncelik:

**Çok yüksek.** Eğitim gerektirmez, mevcut checkpointlerle yapılabilir.

### Work8B: UNetPlusPlus3D İyileştirme Deneyleri

Amaç:

Work8'in kazanan model ailesini küçük, kontrollü deneylerle iyileştirmek.

Ana adaylar:

1. `UNetPlusPlus3D + apbs_shape + apbs_full_signed`
2. `UNetPlusPlus3D + apbs_shape_selected_chem + apbs_full_signed`
3. `UNetPlusPlus3D + apbs_shape_selected_chem + apbs_posneg_clip20`

Denenecek küçük değişkenler:

- Standardizasyon:
  - global standardizasyon
  - kanal bazlı standardizasyon
  - APBS için signed/robust normalizasyon
- Veri artırma:
  - mevcut rotate+flip
  - rotate-only
  - no-augmentation kontrolü
- Kayıp fonksiyonu:
  - mevcut `BCEDiceLoss`
  - `Focal Tversky`
  - `BCE + Tversky`
- Model kapasitesi:
  - `base_features=8`
  - `base_features=12`
  - `base_features=16`

Not:

Bunların hepsini tam matris halinde koşmak gereksiz olur. İlk etapta yalnızca en iyi koşu üzerinde 4-6 küçük deney yeterli.

Öncelik:

**Yüksek.** UNet++ zaten güçlü; küçük ayarlarla 0.7143 Pocket-F1 ve 0.5556 DCC üzerine çıkma ihtimali var.

### Work8C: Literatür Benzeri Model Kontrolü

Amaç:

En iyi öznitelik ve APBS temsilini literatürden esinlenen mimarilerle kısa kontrol etmek.

Denenecek modeller:

- `KalasantyUNet3D`
- `PUResNetV1Like3D`
- `PUResNetV2DenseLike3D`

Opsiyonel:

- `SwinSiteLike3D` yalnızca modern bağlam için; ana tez eksenine koymak zorunda değiliz.

Önerilen özellik:

```text
apbs_shape + apbs_full_signed
```

veya

```text
apbs_shape_selected_chem + apbs_full_signed
```

Neden önemli:

Tez savunmasında "Kalasanty/PUResNet mimarisini denedin mi?" sorusuna kontrollü cevap verir. Bu sonuçlar ana iddia olmak zorunda değil; destekleyici analiz olarak kullanılabilir.

Öncelik:

**Orta.** Ana bilimsel katkı APBS etkisi olduğu için Work8A ve Work8B'den sonra yapılmalı.

## Nihai Öneri

Sıra şu olmalı:

1. Work8A top-k metrik genişletmesini yap.
2. Work8 sonuçlarını top-1/top-3 ayrımıyla yeniden raporla.
3. UNetPlusPlus3D için Work8B küçük iyileştirme deneylerini başlat.
4. Work8C literatür benzeri model kontrolünü kısa ve sınırlı tut.
5. Ardından Work9/Work15 dış veri seti hazırlığı ve değerlendirmesine geç.

Work8'den çıkan ana tez cümlesi:

> APBS elektrostatik potansiyelinin işaretli tam alan temsili, üç boyutlu U-Net++ tabanlı bağlanma bölgesi tahmininde güçlü ve ölçülebilir katkı sağlamış; özellikle APBS+şekil kombinasyonu cep lokalizasyonunda, APBS+şekil+kimyasal öznitelikler ise DCA ve DVO ölçütlerinde öne çıkmıştır.
