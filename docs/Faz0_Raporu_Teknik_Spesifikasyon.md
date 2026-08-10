# FAZ 0 Raporu: Proje Spesifikasyonu ve Deney Altyapısı

## 1. Problem ve Kapsam Tanımı (A-0.1)
NeuroKinematics, endüstriyel robot kollarının ters kinematik (IK) problemini öğrenme tabanlı ve fizik farkındalıklı (physics-aware) yöntemlerle çözmeyi hedefleyen modüler bir platformdur. 

İlk prototip geliştirme ve doğrulama süreçlerinde **KUKA KR6 R900 sixx (6-DoF)** robot modeli "baseline" olarak kullanılacaktır. Diğer robotlar (UR5, ABB IRB 120 vb.) genelleme (generalization) testi aşamasında devreye alınacaktır. MAML veya çoklu robot genellemesine geçilmeden önce KR6 üzerinde tüm modelin (FK → Veri Fabrikası → Baseline MLP → Res-MLP → Physics-Aware Loss → Hybrid Solver) eksiksiz çalıştığı kanıtlanacaktır.

### 1.1. KUKA KR6 R900 sixx — Robot Tanımı ve Kinematik Parametreleri

| Özellik | Değer |
|---|---|
| Üretici | KUKA Robotics |
| Model | KR6 R900 sixx |
| Serbestlik Derecesi (DoF) | 6 (Revolute) |
| Maksimum Erişim | 901 mm |
| Tekrarlama Hassasiyeti | ±0.03 mm |
| Toplam Kütle | ~52 kg |

#### 1.1.1. Denavit-Hartenberg (DH) Parametreleri

| Eklem (i) | $\theta_i$ (rad) | $d_i$ (mm) | $a_i$ (mm) | $\alpha_i$ (rad) |
|---|---|---|---|---|
| 1 | $\theta_1$ | 400 | 25 | $-\pi/2$ |
| 2 | $\theta_2$ | 0 | 315 | 0 |
| 3 | $\theta_3$ | 0 | 35 | $-\pi/2$ |
| 4 | $\theta_4$ | 365 | 0 | $\pi/2$ |
| 5 | $\theta_5$ | 0 | 0 | $-\pi/2$ |
| 6 | $\theta_6$ | 80 | 0 | 0 |

> **Not:** Bu değerler KUKA'nın resmi veri sayfasından alınmış olup, Faz 1'de Pinocchio ve URDF üzerinden yeniden doğrulanacaktır.

#### 1.1.2. Eklem Limitleri

| Eklem | Alt Limit (°) | Üst Limit (°) | Alt Limit (rad) | Üst Limit (rad) |
|---|---|---|---|---|
| $\theta_1$ (A1) | -170 | +170 | -2.9671 | +2.9671 |
| $\theta_2$ (A2) | -190 | +45 | -3.3161 | +0.7854 |
| $\theta_3$ (A3) | -120 | +156 | -2.0944 | +2.7227 |
| $\theta_4$ (A4) | -185 | +185 | -3.2289 | +3.2289 |
| $\theta_5$ (A5) | -120 | +120 | -2.0944 | +2.0944 |
| $\theta_6$ (A6) | -350 | +350 | -6.1087 | +6.1087 |

#### 1.1.3. Genelleme Robotları (Sonraki Fazlar)
| Robot | DoF | Kullanım Amacı |
|---|---|---|
| Universal Robots UR5 | 6 | Genelleme testi (farklı morfoloji) |
| ABB IRB 120 | 6 | Genelleme testi (kompakt yapı) |
| Franka Emika Panda | 7 | İleri aşama: Redundant (fazla DoF) test |

---

### 1.2. Ağın Girdisi (15 Boyut)
Ağ, sadece hedef pozisyonu değil, mevcut eklem durumunu da gözeterek "State-Conditioned" bir yapı sergileyecektir. Bu sayede One-to-Many probleminin çözümü kolaylaşacaktır.

```
                TARGET POSE
        ┌─────────────────────────┐
        │ Position                │
        │ px, py, pz       (3D)  │
        │                         │
        │ 6D Rotation             │
        │ r1 ... r6        (6D)  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ CURRENT ROBOT STATE     │
        │ q(t-1)                  │
        │ θ1 ... θ6        (6D)  │
        └────────────┬────────────┘
                     │
                     ▼
             ┌─────────────┐
             │   Res-MLP    │
             │ Neural IK    │
             │ [15] → [12]  │
             └──────┬──────┘
                    │
                    ▼
          [sin(θ1), cos(θ1),
           sin(θ2), cos(θ2),
           ...
           sin(θ6), cos(θ6)]
```

- **Hedef Pozisyon:** $p_x, p_y, p_z$ (3 Boyut)
- **Hedef Yönelim:** 6D Continuous Rotation Representation $r_1, ..., r_6$ (6 Boyut). Quaternion'un birim norm kısıtı ($||q||=1$) ve antipodal eşdeğerlik ($q \equiv -q$) sorunlarından, Euler açılarının ise gimbal lock ve açı sarmalanması problemlerinden kaçınılmaktadır.
- **Mevcut Robot Durumu:** Bir önceki eklem konfigürasyonu $q_{t-1} = [\theta_1, ..., \theta_6]$ (6 Boyut)
- **Toplam Girdi Boyutu:** 3 + 6 + 6 = **15 Boyut**

### 1.3. Ağın Çıktısı (12 Boyut)
Doğrudan eklem açısı ($\theta$) tahmin etmek yerine periyodiklik kısıtlarını aşmak için sinüs/kosinüs gösterimi kullanılacaktır.

- **Çıktı Formatı:** Her eklem için $[\sin(\theta_i), \cos(\theta_i)]$ olmak üzere toplam **12 Boyut**.
- **Normalizasyon:** Ağın ham çıktısı normalize edilir:
$$\tilde{s}_i = \frac{s_i}{\sqrt{s_i^2 + c_i^2 + \epsilon}}, \quad \tilde{c}_i = \frac{c_i}{\sqrt{s_i^2 + c_i^2 + \epsilon}}$$
- **Gerçek açı:** $\hat{\theta}_i = \text{atan2}(\tilde{s}_i, \tilde{c}_i)$ formülü ile geri dönüştürülür.
- **Motivasyon:** $\theta = 179°$ ile $\theta = -181°$ fiziksel olarak aynı yönelimi ifade eder; ancak MSE açısından çok uzaktır. $(\sin, \cos)$ temsili bu süreksizliği ortadan kaldırır.

### 1.4. Teknik Spesifikasyon Özet Tablosu

| Bileşen | Karar |
|---|---|
| Baseline Robot | KUKA KR6 R900 sixx |
| DoF | 6 |
| Pozisyon Girdisi | $p_x, p_y, p_z$ (3D) |
| Orientation Girdisi | 6D Continuous Rotation (6D) |
| Mevcut Robot Durumu | $q_{t-1}$ (6D) |
| **Toplam Input** | **15 Boyut** |
| Network Mimarisi | Residual MLP (Res-MLP) |
| **Toplam Output** | **12 Boyut** |
| Output Representation | $\sin(\theta_i), \cos(\theta_i)$ |
| Gerçek Eklem Açısı | $\text{atan2}(\sin\theta, \cos\theta)$ |
| FK Doğrulaması | Differentiable FK |
| Fizik Kayıpları | FK + Joint Limit + Singularity + Smoothness + Jerk |
| İlk Aşama | KR6 baseline |
| İkinci Aşama | Çoklu robot (UR5, ABB) |
| İleri Aşama | MAML / Few-Shot |

---

## 2. Başarı Metrikleri (A-0.2)

Deney protokolünde modelin başarısını ölçmek için aşağıdaki **12 metrik** uygulanacaktır:

### 2.1. Doğruluk Metrikleri
| # | Metrik | Tanım | Birim | Fonksiyon |
|---|---|---|---|---|
| 1 | Pozisyon Hatası ($e_p$) | $\|p_{pred} - p_{target}\|_2$ (Öklid mesafesi) | mm | `compute_position_error()` |
| 2 | Yönelim Hatası ($e_R$) Geodesic | $\arccos\left(\frac{\text{tr}(R_{pred}^T R_{target}) - 1}{2}\right)$ | rad | `compute_orientation_error_geodesic()` |
| 3 | Yönelim Hatası ($e_R$) 6D L2 | $\|r_{pred} - r_{target}\|_2$ (6D uzayda) | - | `compute_orientation_error_6d()` |
| 4 | IK Başarı Oranı @1mm | $e_p < 1\text{mm}$ olan örneklerin yüzdesi | % | `compute_ik_success_rate(threshold=1.0)` |
| 5 | IK Başarı Oranı @5mm | $e_p < 5\text{mm}$ olan örneklerin yüzdesi | % | `compute_ik_success_rate(threshold=5.0)` |

### 2.2. Fiziksel Kısıt Metrikleri
| # | Metrik | Tanım | Birim | Fonksiyon |
|---|---|---|---|---|
| 6 | Joint-Limit İhlal Oranı | Tahmin edilen $\theta_i$'nin $[q_{min}, q_{max}]$ dışında kalan miktarı | rad | `compute_joint_limit_violation()` |
| 7 | Tekillik Bölgesi Başarı Oranı | Manipulability $< 0.01$ olan pozlarda IK doğruluğu | % | `compute_singularity_success_rate()` |

### 2.3. Yörünge Kalitesi Metrikleri
| # | Metrik | Tanım | Birim | Fonksiyon |
|---|---|---|---|---|
| 8 | Jitter | Ardışık eklem açısı farklarının standart sapması | rad | `compute_jitter()` |
| 9 | Yörünge Sürekliliği (Smoothness) | İvme profilinin (2. türev) normu | rad/s² | `compute_smoothness()` |
| 10 | Jerk (Sarsıntı) | 3. türevin normu | rad/s³ | `compute_jerk()` |

### 2.4. Performans ve Kaynak Metrikleri
| # | Metrik | Tanım | Birim | Fonksiyon |
|---|---|---|---|---|
| 11 | Inference Latency | Ortalama, P95, P99 çıkarım süreleri | ms | `compute_inference_latency()` |
| 12 | Kaynak Tüketimi | CPU %, RAM MB, GPU MB, GPU % | çeşitli | `compute_resource_usage()` |

---

## 3. Baseline Deney Protokolü (A-0.3)

### 3.1. Test Edilecek Modeller
| # | Model | Tip | Açıklama |
|---|---|---|---|
| 1 | KDL | Sayısal (Iteratif) | Jacobian tabanlı klasik IK çözücü |
| 2 | TRAC-IK | Sayısal (Hibrit) | KDL + SQP eşzamanlı, tekilliklere dirençli |
| 3 | Baseline MLP | Nöral (Fizik-Unaware) | Sadece MSE kaybı, fizik terimi yok |
| 4 | Res-MLP (Bizim) | Nöral (Physics-Aware) | FK + Joint Limit + Singularity + Smoothness + Jerk kayıpları |

### 3.2. Deney Seti Oluşturma Yöntemi
- Veri kümesi toplam **1.000.000 sentetik örnek** içerecektir (FK üzerinden üretilecek).
- Veri bölümleme: **%80 Train / %10 Validation / %10 Test** (100.000 test örneği).
- Bölümleme, **uzaysal (spatial) tabanlı** yapılacaktır: Test seti, eğitim setinde bulunmayan çalışma alanı bölgelerini de içerecektir (data leakage önlemi).
- Tüm modeller **aynı test seti** üzerinde değerlendirilecektir.

### 3.3. İstatistiksel Güvenilirlik Protokolü
- Her nöral model **5 farklı random seed** ile ($S = \{42, 123, 456, 789, 1024\}$) eğitilecektir.
- Sonuçlar ortalama ± standart sapma olarak raporlanacaktır.
- KDL ve TRAC-IK deterministik olduğundan tek çalıştırma yeterlidir.
- Inference latency ölçümü **100 warm-up + 1000 ölçüm** üzerinden yapılacak, P95 ve P99 percentile raporlanacaktır.

### 3.4. Eşit Koşul Garantileri
- Tüm modellere **aynı hedef pozlar** verilecektir.
- KDL ve TRAC-IK için başlangıç konfigürasyonu ($q_0$), nöral modeldeki $q_{t-1}$ ile **aynı** olacaktır.
- Latency ölçümleri **aynı donanım** üzerinde yapılacaktır.
- Donanım ortamı: Python sürümü, PyTorch sürümü, CUDA sürümü, GPU/CPU modeli raporlanacaktır.

### 3.5. Sonuç Raporu Formatı
Her deney sonucunda aşağıdaki tablo doldurulacaktır:

| Metrik | KDL | TRAC-IK | Baseline MLP | Res-MLP (Ours) |
|---|---|---|---|---|
| Ort. $e_p$ (mm) | - | - | - ± - | - ± - |
| Maks. $e_p$ (mm) | - | - | - ± - | - ± - |
| Ort. $e_R$ (rad) | - | - | - ± - | - ± - |
| IK Başarı @1mm (%) | - | - | - ± - | - ± - |
| IK Başarı @5mm (%) | - | - | - ± - | - ± - |
| Joint-Limit İhlal (%) | - | - | - ± - | - ± - |
| Tekillik Başarı (%) | - | - | - ± - | - ± - |
| Ort. Latency (ms) | - | - | - | - |
| P99 Latency (ms) | - | - | - | - |

---

## 4. Geliştirme Yol Haritası Özeti
**Slogan:** Önce KR6 Baseline → Sonra Genelleme → Sonra Ürünleştirme.

```
KR6 URDF → FK → Jacobian → Workspace → Synthetic Dataset
    → Baseline MLP → Res-MLP → Physics-Aware Loss → Hybrid Solver
        → Çoklu Robot Genellemesi (UR5, ABB) → MAML / Edge AI
```
