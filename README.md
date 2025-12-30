# resnet_bone_age_prediction

## Proje Konusu
El röntgeni görüntülerinden kemik yaşı tahmini yapan derin öğrenme modeli. ResNet50 ve transfer learning kullanarak RSNA 2017 dataseti üzerinde eğitildi.
## Seçilme Gerekçesi
Kemik yaşı tahmini, çocuklarda büyüme bozuklukları, hormonal hastalıklar ve gelişimsel gecikmelerin teşhisinde kritik öneme sahiptir. Pediatrik endokrinolojide rutin olarak kullanılır, adli tıpta yaş tespitinde ve genetik hastalıkların erken teşhisinde önemli rol oynar. Geleneksel yöntemler (Greulich-Pyle, Tanner-Whitehouse) radyologların manuel değerlendirmesine dayanır ve subjektif farklılıklara açıktır. Derin öğrenme ile bu süreç otomatikleştirilerek hem hız hem de tutarlılık sağlanabilir.

## Alanda Yapılan Çalışmalar

| Çalışma | Yıl | Model | Dataset | MAE | Detay |
|---------|-----|-------|---------|-----|-------|
| **RSNA 2017** | 2017 | Ensemble | RSNA (12,600) | 4.2 ay | 200+ takım, en iyi sonuç ensemble ile |
| **Lee et al.** | 2017 | GoogLeNet | Özel (8,000) | 10 ay | Güney Kore hastane verileri |
| **Larson et al.** | 2018 | Custom CNN | RSNA | 6.2 ay | Stanford Üniversitesi çalışması |
| **Bu Proje** | 2025 | ResNet50 | RSNA | **8.7 ay** | Tek model, 10 epoch |


## Model Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUT                               │
│         El röntgeni (224x224 RGB) + Cinsiyet (0/1)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   ResNet50 Backbone                         │
│              (ImageNet pretrained, 26M param)               │
│                    → 2048 features                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Gender Embedding                           │
│              Linear(1→32) + ReLU + Dropout(0.3)             │
│                    → 32 features                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Regressor Head                            │
│    Linear(2080→1024) + BN + ReLU + Dropout(0.4)             │
│    Linear(1024→512) + BN + ReLU + Dropout(0.3)              │
│    Linear(512→1)                                            │
│                    → Kemik yaşı (ay)                        │
└─────────────────────────────────────────────────────────────┘
```

## Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| **Framework** | PyTorch |
| **Model** | ResNet50 (Pretrained) |
| **Katman Sayısı** | 50 |
| **Parametre Sayısı** | 26,167,425 |
| **Görüntü Boyutu** | 224×224 |
| **Renk Kanalı** | RGB (3) |
| **Batch Size** | 16 |
| **Epochs** | 10 |
| **Learning Rate** | 0.0001 (1e-4) |
| **Optimizer** | AdamW |
| **Weight Decay** | 1e-5 |
| **Loss Function** | MSE |
| **Validation Split** | %15 |
| **Target Normalization** | Yok (doğrudan ay) |
| **Pretrained** | ✅ ImageNet |
| **Gender Input** | ✅ Kullanıyor |
| **LR Scheduler** | ReduceLROnPlateau (patience=2) |
| **Early Stopping** | Yok |

## Dataset
[https://www.kaggle.com/datasets/kmader/rsna-bone-age](https://www.kaggle.com/datasets/kmader/rsna-bone-age)

RSNA = Radiological Society of North America

## Kurulum

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas pillow tqdm numpy matplotlib gradio
```

## Çalıştırma

```powershell
python train.py   # Model eğitimi
python test.py    # Test ve değerlendirme
python app.py     # (İsteğe bağlı) Gradio demo arayüzü
```