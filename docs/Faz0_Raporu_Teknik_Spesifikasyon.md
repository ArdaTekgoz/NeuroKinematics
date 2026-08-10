# FAZ 0 Raporu: Proje Spesifikasyonu ve Deney Altyapısı

## 1. Problem ve Kapsam Tanımı (A-0.1)
NeuroKinematics, endüstriyel robot kollarının ters kinematik (IK) problemini öğrenme tabanlı ve fizik farkındalıklı (physics-aware) yöntemlerle çözmeyi hedefleyen modüler bir platformdur. 

İlk prototip geliştirme ve doğrulama süreçlerinde **KUKA KR6 (6-DoF)** robot modeli "baseline" olarak kullanılacaktır. Diğer robotlar (UR5, ABB IRB 120 vb.) genelleme (generalization) testi aşamasında devreye alınacaktır. MAML veya çoklu robot genellemesine geçilmeden önce KR6 üzerinde tüm modelin (FK → Veri Fabrikası → Baseline MLP → Res-MLP → Physics-Aware Loss → Hybrid Solver) eksiksiz çalıştığı kanıtlanacaktır.

### 1.1. Ağın Girdisi (15 Boyut)
Ağ, sadece hedef pozisyonu değil, mevcut eklem durumunu da gözeterek "State-Conditioned" bir yapı sergileyecektir. Bu sayede One-to-Many probleminin çözümü kolaylaşacaktır.
- **Hedef Pozisyon:** $p_x, p_y, p_z$ (3 Boyut)
- **Hedef Yönelim:** 6D Continuous Rotation Representation $r_1, ..., r_6$ (6 Boyut). Quaternion veya Euler açılarının süreksizlik/kısıt problemleri bu sayede aşılmıştır.
- **Mevcut Robot Durumu:** Bir önceki eklem konfigürasyonu $q_{t-1} = [\theta_1, ..., \theta_6]$ (6 Boyut)
- **Toplam Girdi Boyutu:** 3 + 6 + 6 = 15 Boyut.

### 1.2. Ağın Çıktısı (12 Boyut)
Doğrudan eklem açısı ($\theta$) tahmin etmek yerine periyodiklik kısıtlarını aşmak için sinüs/kosinüs gösterimi kullanılacaktır.
- **Çıktı Formatı:** Her eklem için $[\sin(\theta_i), \cos(\theta_i)]$ olmak üzere toplam 12 Boyut.
- **Normalizasyon:** Ağın çıktısı normalize edilerek ($\tilde{s}_i, \tilde{c}_i$) gerçek eklem açısı $\hat{\theta}_i = \text{atan2}(\tilde{s}_i, \tilde{c}_i)$ formülü ile geri dönüştürülecektir.

## 2. Başarı Metrikleri (A-0.2)
Deney protokolünde ağın başarısını ölçmek için aşağıdaki metrikler uygulanacaktır:
1. **Pozisyon Hatası ($e_p$):** Kartezyen uzayda Öklid mesafesi.
2. **Yönelim Hatası ($e_R$):** Rotasyonel fark.
3. **IK Başarı Oranı:** Hatanın belirli bir toleransın (örn. 1mm) altında olma yüzdesi.
4. **Joint-Limit İhlal Oranı:** Tahmin edilen $\theta$'nın limitler dışında kalma durumu.
5. **Jitter ve Yörünge Sürekliliği:** Peş peşe gelen hareketlerde eklem açısı zıplamaları.
6. **Inference Latency & P95/P99:** Tahmin süresinin milisaniye cinsinden donanım profillemesi.

## 3. Baseline Protokolü (A-0.3)
Geliştirilecek olan Physics-Aware Res-MLP modelinin sonuçları, aşağıdaki solver'lar ile aynı test setinde karşılaştırılacaktır:
- Analitik/Sayısal Referanslar: KDL, TRAC-IK
- Nöral Baseline: Standart MLP (Fizik kayıpları olmadan)

## 4. Geliştirme Yol Haritası Özeti
**Slogan:** Önce KR6 Baseline → Sonra Genelleme → Sonra Ürünleştirme.
1. KR6 URDF Okuma
2. Differentiable FK ve Jacobian
3. Workspace Analizi ve Sentetik Veri
4. Baseline MLP Eğitimi
5. Res-MLP ve Physics-Aware Loss 
6. Hybrid Solver (Nöral + Sayısal Optimizasyon)
7. Çoklu Robot Genellemesi (UR5, ABB)
8. MAML / Edge AI Deployment
