# 🔬 Kemik Yaşı Tahmininde Cinsiyet Etkisi Analizi

## 📊 Training Set Yaş Dağılımı

| Yaş Grubu | Örnek Sayısı | Yüzde |
|-----------|-------------|-------|
| 0-4 yaş | 482 | %3.9 |
| 4-8 yaş | 2,395 | %19.5 |
| 8-12 yaş | 4,119 | %33.5 |
| 12-16 yaş | 4,813 | %39.1 |
| 16-18 yaş | 412 | %3.3 |
| 18-20 yaş | 90 | %0.7 |
| 20+ yaş | 0 | %0 |
| **Toplam** | **12,311** | **100%** |

> [!IMPORTANT]
> Veri setinin **%72.6'sı 8-16 yaş** (ergenlik dönemi) aralığında. 18+ yaş grubunda sadece **90 örnek** var.

---

## 📈 Model Gender Farkı Sonuçları

Aynı görüntü için erkek vs kadın seçildiğinde tahmin farkı:

| Yaş Grubu | Örnek | Erkek-Kadın Farkı |
|-----------|-------|-------------------|
| 0-4 yaş | 15 | +8.4 ay |
| 4-8 yaş | 56 | +16.0 ay |
| 8-10 yaş | 31 | +21.2 ay |
| **10-12 yaş** | **65** | **+23.1 ay** ← Zirve |
| 12-14 yaş | 80 | +21.2 ay |
| 14-16 yaş | 37 | +19.2 ay |
| 16-18 yaş | 13 | +17.7 ay |
| 18+ yaş | 3 | +19.1 ay |

---

## 🔍 Neden Bu Fark Var?

### Biyolojik Gerçeklik
- **Kızlar ergenliğe erkeklerden ~1-2 yıl önce girer**
- Aynı kronolojik yaştaki bir kız, erkekten daha olgun kemik yapısına sahip
- Bu yüzden aynı röntgen görüntüsü için:
  - Erkek seçildiğinde → daha yüksek kemik yaşı
  - Kadın seçildiğinde → daha düşük kemik yaşı

### Dataset Etkisi
Training setinin **%72.6'sı ergenlik döneminde** olduğu için model bu farkı çok iyi öğrenmiş.

---

## ⚠️ Fark Neden 18+ Yaşta Kapanmıyor?

### Gerçek Hayatta Ne Olmalı?
- 18 yaş sonrası **kemik büyümesi durur**
- Hem erkek hem kadın maksimum kemik olgunluğuna ulaşır
- **Gender farkı sıfıra yaklaşmalı**

### Model Neden Bunu Öğrenemedi?
- Training setinde 18+ yaş sadece **90 örnek** (%0.7)
- Model bu yaş grubunu yeterince görmemiş
- Ergenlik dönemindeki pattern'i yetişkinlere de uyguluyor

---

## 💡 Sonuç

| Bulgu | Açıklama |
|-------|----------|
| ✅ Model doğru öğrenmiş | Ergenlik döneminde cinsiyet farkını yakalıyor |
| ⚠️ Dataset dengesiz | Yetişkin örneği çok az |
| ❌ Fark kapanmıyor | 18+ yaşta hala ~19 ay fark var |

**Çözüm:** Model 18+ yaşta farkın kapandığını öğrenmesi için training setinde daha fazla yetişkin örneği gerekli.
