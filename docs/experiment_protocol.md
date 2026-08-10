# NeuroKinematics - Deney Protokolü ve Baseline Karşılaştırma Altyapısı

## 1. Amaç (A-0.3)
NeuroKinematics'in geliştireceği "Fizik Farkındalıklı" (Physics-Aware) Nöral Ters Kinematik çözücüsü olan **Res-MLP** modelinin performansının, aynı veri ve test koşulları altında mevcut literatürdeki referans çözücülerle (baseline) objektif olarak karşılaştırılması amaçlanmaktadır.

## 2. Test Edilecek Baseline Modeller

### 2.1. KDL (Kinematics and Dynamics Library)
- **Tip:** Klasik Jacobian tabanlı iteratif IK çözücü.
- **Çalışma Prensibi:** Jacobian pseudo-inverse yöntemiyle $q_{k+1} = q_k + J(q_k)^\dagger e_k$ iterasyonları.
- **Başlangıç Konfigürasyonu:** Nöral modeldeki $q_{t-1}$ ile aynı.
- **Karşılaştırma Amacı:** Klasik yöntemin doğruluk ve hız referansını ortaya koymak.

### 2.2. TRAC-IK
- **Tip:** Hibrit sayısal çözücü (Jacobian + SQP eşzamanlı).
- **Çalışma Prensibi:** KDL'nin iteratif yaklaşımı ile SQP (Sequential Quadratic Programming) optimizasyonunu paralel çalıştırır; ilk bulan kazanır.
- **Başlangıç Konfigürasyonu:** Nöral modeldeki $q_{t-1}$ ile aynı.
- **Karşılaştırma Amacı:** Tekilliklere dirençli en iyi sayısal çözücü referansı.

### 2.3. Baseline MLP (Physics-Unaware)
- **Tip:** Standart çok katmanlı algılayıcı (MLP).
- **Mimari:** Res-MLP ile aynı katman boyutları ([512, 512, 512, 256]), ancak residual bağlantı YOK.
- **Kayıp Fonksiyonu:** Sadece MSE (Mean Squared Error) kaybı. FK, Joint Limit, Singularity gibi fizik terimleri DAHİL EDİLMEZ.
- **Girdi/Çıktı:** Res-MLP ile aynı (15 → 12).
- **Karşılaştırma Amacı:** Physics-Aware yaklaşımının (FK loss, joint limit penalty vb.) gerçek etkisini izole etmek.

### 2.4. Res-MLP — NeuroKinematics Physics-Aware Model (Bizim Modelimiz)
- **Tip:** Residual MLP mimarisi.
- **Mimari:** [512, 512, 512, 256] boyutlu gizli katmanlar + residual (skip) bağlantılar.
- **Kayıp Fonksiyonu:** Çok bileşenli fizik farkındalıklı kayıp:
  - MSE Loss (eklem uzayında)
  - Differentiable FK Position Loss
  - Differentiable FK Rotation Loss
  - Joint Limit Penalty
  - Singularity Penalty (Yoshikawa manipulability)
  - Smoothness / Jerk terimi
- **Girdi:** $[p_x, p_y, p_z, r_1...r_6, \theta_1^{t-1}...\theta_6^{t-1}]$ → 15 boyut
- **Çıktı:** $[\sin\theta_1, \cos\theta_1, ..., \sin\theta_6, \cos\theta_6]$ → 12 boyut

---

## 3. Deney Seti Oluşturma Yöntemi

### 3.1. Veri Boyutu ve Kaynağı
- **Kaynak:** Sentetik veri; eklem uzayında rastgele örneklenen $q$ konfigürasyonlarının Forward Kinematics ile Kartezyen poza dönüştürülmesi.
- **Toplam Örnek:** 1.000.000
- **Robot:** KUKA KR6 R900 sixx (6-DoF)

### 3.2. Veri Bölümleme
| Bölüm | Oran | Örnek Sayısı | Açıklama |
|---|---|---|---|
| Train | %80 | 800.000 | Model eğitimi |
| Validation | %10 | 100.000 | Hiperparametre seçimi ve erken durdurma (early stopping) |
| Test | %10 | 100.000 | Nihai performans değerlendirmesi |

### 3.3. Data Leakage Önlemi
- Bölümleme **uzaysal (spatial) tabanlı** yapılacaktır.
- Test seti, eğitim setinde bulunmayan çalışma alanı bölgelerini de içerecektir.
- Aynı $q$ konfigürasyonunun hem eğitim hem test setinde bulunması engellenecektir.

---

## 4. İstatistiksel Güvenilirlik Protokolü

### 4.1. Tekrar Sayısı ve Seed'ler
| Model | Deterministik mi? | Tekrar Sayısı | Seed'ler |
|---|---|---|---|
| KDL | Evet | 1 | — |
| TRAC-IK | Evet | 1 | — |
| Baseline MLP | Hayır | 5 | {42, 123, 456, 789, 1024} |
| Res-MLP | Hayır | 5 | {42, 123, 456, 789, 1024} |

### 4.2. Raporlama Formatı
- Nöral modeller: **Ortalama ± Standart Sapma** olarak raporlanacak.
- Deterministik modeller: **Tek değer** olarak raporlanacak.
- Tüm metrikler hem ortalama (mean) hem de medyan (median) değerleri ile sunulacak.

### 4.3. Inference Latency Ölçüm Protokolü
- **Warm-up:** İlk 100 çıkarım sonucu dikkate alınmaz (GPU/CPU cache ısınması).
- **Ölçüm:** Sonraki 1000 çıkarım üzerinden istatistik alınır.
- **Raporlanan:** Ortalama, Standart Sapma, P95, P99, Min, Max (milisaniye cinsinden).
- **GPU Zamanlama:** Eğer GPU mevcutsa `torch.cuda.Event` ile senkronize zamanlama yapılır.
- **CPU Zamanlama:** `time.perf_counter()` (yüksek çözünürlüklü monoton saat) kullanılır.

---

## 5. Eşit Koşul Garantileri

Karşılaştırmanın bilimsel geçerliliğini sağlamak için aşağıdaki koşullar garanti altına alınacaktır:

| Koşul | Detay |
|---|---|
| Aynı Test Seti | 4 modelin tamamı aynı 100.000 test pozunu çözecektir |
| Aynı Başlangıç Konfigürasyonu | KDL/TRAC-IK'nın $q_0$'ı = Nöral modeldeki $q_{t-1}$ |
| Aynı Donanım | Tüm latency ölçümleri aynı makine üzerinde yapılacaktır |
| Ortam Raporu | Python, PyTorch, CUDA, CPU/GPU modeli, RAM raporlanacaktır |
| Aynı Tolerans | IK başarı oranı aynı $\epsilon$ eşik değerleri ile hesaplanacaktır |

---

## 6. Sonuç Raporu Tablosu (Benchmark Çıktısı)

Her deney sonucunda aşağıdaki tablo doldurularak `experiments/` dizinine kaydedilecektir:

| Metrik | KDL | TRAC-IK | Baseline MLP | Res-MLP (Ours) |
|---|---|---|---|---|
| Ort. $e_p$ (mm) | — | — | — ± — | — ± — |
| Medyan $e_p$ (mm) | — | — | — ± — | — ± — |
| Maks. $e_p$ (mm) | — | — | — ± — | — ± — |
| Ort. $e_R$ geodesic (rad) | — | — | — ± — | — ± — |
| IK Başarı @1mm (%) | — | — | — ± — | — ± — |
| IK Başarı @5mm (%) | — | — | — ± — | — ± — |
| Joint-Limit İhlal (%) | — | — | — ± — | — ± — |
| Tekillik Başarı (%) | — | — | — ± — | — ± — |
| Jitter (rad) | — | — | — ± — | — ± — |
| Smoothness (rad/s²) | — | — | — ± — | — ± — |
| Jerk (rad/s³) | — | — | — ± — | — ± — |
| Ort. Latency (ms) | — | — | — | — |
| P95 Latency (ms) | — | — | — | — |
| P99 Latency (ms) | — | — | — | — |
| CPU (%) | — | — | — | — |
| RAM (MB) | — | — | — | — |
| GPU Mem (MB) | — | — | — | — |
