# Model Mimari Diyagramları

Bu SVG dosyaları Deep-APBS deneylerinde kullanılan veya denenmesi planlanan ana 3B mimariler için tezde kullanılabilir kavramsal diyagramlardır.
Amaç okunması zor PyTorch hesap grafiği dökmek değil, mimari ailelerinin çalışma mantığını anlaşılır şekilde özetlemektir.

| Dosya | Model | Tezde kullanım |
|---|---|---|
| `unet3d4l.svg` | UNet3D4L | Voksel düzeyinde cep segmentasyonu için temel 4 seviyeli 3B U-Net kodlayıcı-çözücü |
| `unet3d4la.svg` | UNet3D4LA | Çözümleme öncesinde atlama özniteliklerini süzen dikkat kapılı 4 seviyeli 3B U-Net |
| `resnet3d4l.svg` | ResNet3D4L | BasicBlock3D modülleri ve U-Net benzeri atlama bağlantıları kullanan artık 3B kodlayıcı-çözücü |
| `unetplusplus3d.svg` | UNetPlusPlus3D | Kodlayıcı-çözücü anlam farkını azaltan yoğun atlama yollarına sahip iç içe 3B U-Net |
| `cbamunet3d.svg` | CBAMUNet3D | Her blokta CBAM kanal ve uzamsal dikkat kullanan artık 3B U-Net |
| `resnet3d4lgn.svg` | ResNet3D4LGN | Küçük batch koşullarında kararlılık için GroupNorm kullanan ResNet3D4L tarzı mimari |
| `tinyconvnextunet3d.svg` | TinyConvNeXtUNet3D | Depthwise konvolüsyon blokları kullanan hafif ConvNeXt tarzı 3B U-Net |
| `kalasantyunet3d.svg` | KalasantyUNet3D | 2, 2, 3, 3 havuzlama düzenine sahip Kalasanty tarzı 3B U-Net |
| `puresnetv1like3d.svg` | PUResNetV1Like3D | PUResNet v1 artık kodlayıcı-çözücü topolojisinin dense PyTorch yaklaşımı |
| `puresnetv2denselike3d.svg` | PUResNetV2DenseLike3D | PUResNetV2 sparse artık kodlayıcı-çözücüsünün dense yaklaşımı |
| `swinsitelike3d.svg` | SwinSiteLike3D | Uzak bağlamı yakalamak için transformer tarzı darboğaz kullanan hibrit CNN kodlayıcı-çözücü |

Önerilen kullanım:

- Bölüm 3: `unet3d4l.svg` genel 3B U-Net anlatımı için kullanılabilir.
- Bölüm 4: gerçek deneylerde kullanılan Work8 modelleri için seçili diyagramlar kullanılabilir.
- Literatür karşılaştırması: `kalasantyunet3d.svg`, `puresnetv1like3d.svg` ve `swinsitelike3d.svg` sadece bu modeller nihai yöntem karşılaştırmasında tartışılırsa kullanılmalı.

Tüm diyagramlarda okunabilirlik için temsili `36^3` giriş gridi gösterilmiştir. Aynı topoloji bellek izin verdiğinde daha büyük gridlerle de kullanılabilir.
