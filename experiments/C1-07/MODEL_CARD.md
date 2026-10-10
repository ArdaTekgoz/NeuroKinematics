# Araştırma modeli kartı — nihai Core devri

11 Ekim 2026 · Belge r2 · Yazılım hedefi v1.0.0.
FINAL / RESEARCH_ONLY / DIRECT_IK_NO_GO / G1_RESEARCH_ACCEPTED.
T-C06 temiz ortam doğrulaması PASS; G1 araştırma kapanışı kabul edildi.

## Kimlik ve amaç

KUKA KR6 R900 sixx için çevrimdışı IK araştırması. FK_TANH ana araştırma
adayı; LOCAL_RAW ek yerel karşılaştırma adayı. Amaç gelecekte aynı sayısal
çözücüde neural seed faydasını ölçmektir; mevcut hibrit fayda ölçümü yok.
Fiziksel hareket, collision-free veya güvenlik sertifikası kapsamı yok.

Robot `kuka_kr6_r900_sixx`; base `base_link`; tip `flange`; TCP `tool0`.
Joint sırası: joint_1, joint_2, joint_3, joint_4, joint_5, joint_6. Eklemler rad, konum m; quaternion canonical unit wxyz.

| Girdi varlığı | SHA256 |
|---|---|
| assets/robots/robot_a/robot.urdf | `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96` |
| assets/robots/robot_a/robot_spec.json | `4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d` |
| assets/robots/robot_a/manifest.json | `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1` |
| config/robots/tcp_tool0.json | `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18` |

Eklem limitleri (rad, inclusive):

| Eklem | Alt | Üst |
|---|---:|---:|
| joint_1 | -2.96705972839 | 2.96705972839 |
| joint_2 | -3.31612557879 | 0.785398163397 |
| joint_3 | -2.09439510239 | 2.72271363311 |
| joint_4 | -3.22885911619 | 3.22885911619 |
| joint_5 | -2.09439510239 | 2.09439510239 |
| joint_6 | -6.10865238198 | 6.10865238198 |

## Veri ve eğitim

FK_TANH: C1-02/v1 train16800 satır, label_present15204; local+wide ve
main/boundary/singularity. Eksik1596 wide label supervised paydasına
alınmaz; validation3600 tam paydadır (3249 etiket,351 eksik wide).
Target teacher eklemleri eğitim içindir; çıkarımda yalnız pose ve current
kullanılır. Eğitim kaybı/optimizer/bütçe/seed kuralları manifestteki
experiments/C1-05/config.json ve checkpoint metadata ile sabittir.
Checkpoint best validation Q-loss; üç seed'in seçilmiş epoch'ları7/186/8.
Bu farklı gerçekleşmiş eğitim süreleri adil yeni eş-bütçe eğitimi değildir.

LOCAL_RAW: ADR-021 directions verisi512 main train kökü×8 yön=4096;
original+7 yeni yön, ayrıca bağımsız8 probe yön/root. Hareketler±.1rad
joint farkı, limit dışı vektörler yeniden örneklenir. Üç seed5000 full-batch
AdamW adımında terminal checkpoint; width512,lr.001,wd.01,cosineeta1e-6.
Normalization diagnostic3'ün mevcut train-only göreli konum istatistiğidir;
512 kökten yeniden fit edilmedi. Wide,zor-family eğitim kapsamı yok.
Eğitim seed/data üretim kimliği diagnostic6/9 ön kayıtlarına bağlıdır.

Her iki aile için validation araştırma/izleme amacıyla kullanılmıştır.
Yeni C1-06R bağımsız final NOT_CREATED. Eski C1-06'da yayımlanmış negatif
sonuçlar korunur; eski final raw bu hazırlıkta okunmadı. Aynı sorguların
birden fazla modelde ölçülmesi bağımsız hedef sayısını artırmaz.

## Giriş ve çıkış sözleşmeleri

FK_TANH13 giriş: absolute target p'nin train mean/std ile ölçeği3,
canonical target wxyz4,current eklemlerinin inclusive affine ölçeği6.
MLP13→256→256→256→6 SiLU, çıkış z=(tanh(logits)+1)/2;
q=lower+float64(z)*(upper-lower). Tarihsel c105.infer yolu değiştirilmedi.

LOCAL_RAW13 giriş: p_target-p_current BASE eksenlerinde, train mean/std
ile ölçekli3; R_current.T@R_target CURRENT TCP eksenlerinde canonical
quaternion4; normalize current6. MLP13→512→512→512→6 SiLU;
z=current_norm+MLP(x), fiziksel dönüşüm endpoint-exact float64.
Çıkış bounded değildir, clamp uygulanmaz. Göreli feature için current FK
maliyeti gelecekte toplam inference/solver bütçesine dahil edilmelidir.

İki ailede teacher q_target, root etiketi, family/mode/split/group/pair ID
model girdisi değildir. Aynı boyuttaki iki13-vector sözleşmesi birbirinin
yerine kullanılamaz; model ailesiyle doğru scaler/feature/decoder eşlenir.

## Sabitlenmiş checkpointler

### FK_TANH-2026100201

Rol: `PRIMARY_RESEARCH_CANDIDATE`. Parametre sayısı: 136710.

- checkpoint: `data/generated/C1-05/v1/E-C05/seed-2026100201/attempt-001/FK_TANH/best.pt`
  SHA256: `29d0f2c38ebdd422429fc327e10c4420f77193077e41e4b247a541b5bbf047d9`
- normalization: `experiments/C1-02/normalization.json`
  SHA256: `8508c26a4a42b076be1c3ab88434236a38e21dac59981a1206e2b00d9bbedc86`
- source_config: `experiments/C1-05/config.json`
  SHA256: `ef3c9097a2603a511c2266e80963b9ae1e5deff14cf27ca6aa580c993cd58775`

Bu hazırlık CPU validation: A0/3600, B0/3600; limit dışı0. Local geçersiz0/1800; wide geçersiz0/1800.

### FK_TANH-2026100202

Rol: `PRIMARY_RESEARCH_CANDIDATE`. Parametre sayısı: 136710.

- checkpoint: `data/generated/C1-05/v1/E-C05/seed-2026100202/attempt-001/FK_TANH/best.pt`
  SHA256: `a45b72999ad24e421c4b581859666ae8d4e7bee58b1079c49fc8709e83b763ad`
- normalization: `experiments/C1-02/normalization.json`
  SHA256: `8508c26a4a42b076be1c3ab88434236a38e21dac59981a1206e2b00d9bbedc86`
- source_config: `experiments/C1-05/config.json`
  SHA256: `ef3c9097a2603a511c2266e80963b9ae1e5deff14cf27ca6aa580c993cd58775`

Bu hazırlık CPU validation: A0/3600, B0/3600; limit dışı0. Local geçersiz0/1800; wide geçersiz0/1800.

### FK_TANH-2026100203

Rol: `PRIMARY_RESEARCH_CANDIDATE`. Parametre sayısı: 136710.

- checkpoint: `data/generated/C1-05/v1/E-C05/seed-2026100203/attempt-001/FK_TANH/best.pt`
  SHA256: `486cbece1ff553aec421ece9101b177f079abbd283bda070830b2fbf52f8f069`
- normalization: `experiments/C1-02/normalization.json`
  SHA256: `8508c26a4a42b076be1c3ab88434236a38e21dac59981a1206e2b00d9bbedc86`
- source_config: `experiments/C1-05/config.json`
  SHA256: `ef3c9097a2603a511c2266e80963b9ae1e5deff14cf27ca6aa580c993cd58775`

Bu hazırlık CPU validation: A0/3600, B0/3600; limit dışı0. Local geçersiz0/1800; wide geçersiz0/1800.

### LOCAL_RAW-2026100901

Rol: `EXPLORATORY_LOCAL_ONLY_NOT_PRIMARY_H1`. Parametre sayısı: 535558.

- checkpoint: `data/generated/C1-06R/diagnostic9/s2026100901-RAW/last.pt`
  SHA256: `db1bb987a7f7a69a275bd1558531459cf249cdf5a53b462c9df1b6c1194c1702`
- normalization: `experiments/C1-06R/diagnostic3/normalization.json`
  SHA256: `c751482c75703056d20717773b2983cf6928e8ee28d726bcc275bdd31a92b057`
- source_config: `experiments/C1-06R/diagnostic9/config.json`
  SHA256: `a782486c531f99a00247702e6aae8ffc74c7239cc64f776e1fd7679c41962301`

Bu hazırlık CPU validation: A0/3600, B0/3600; limit dışı1417. Local geçersiz63/1800; wide geçersiz1354/1800.

### LOCAL_RAW-2026100902

Rol: `EXPLORATORY_LOCAL_ONLY_NOT_PRIMARY_H1`. Parametre sayısı: 535558.

- checkpoint: `data/generated/C1-06R/diagnostic9/s2026100902-RAW/last.pt`
  SHA256: `2fe29c962bacf168476896ae093dd5b4384e36a1c25b2b6d0961d8064a4584ab`
- normalization: `experiments/C1-06R/diagnostic3/normalization.json`
  SHA256: `c751482c75703056d20717773b2983cf6928e8ee28d726bcc275bdd31a92b057`
- source_config: `experiments/C1-06R/diagnostic9/config.json`
  SHA256: `a782486c531f99a00247702e6aae8ffc74c7239cc64f776e1fd7679c41962301`

Bu hazırlık CPU validation: A0/3600, B0/3600; limit dışı1404. Local geçersiz73/1800; wide geçersiz1331/1800.

### LOCAL_RAW-2026100903

Rol: `EXPLORATORY_LOCAL_ONLY_NOT_PRIMARY_H1`. Parametre sayısı: 535558.

- checkpoint: `data/generated/C1-06R/diagnostic9/s2026100903-RAW/last.pt`
  SHA256: `10108894757312445abe6ae572015c5fc04fbc5c4f81b3e6b1505b037d811874`
- normalization: `experiments/C1-06R/diagnostic3/normalization.json`
  SHA256: `c751482c75703056d20717773b2983cf6928e8ee28d726bcc275bdd31a92b057`
- source_config: `experiments/C1-06R/diagnostic9/config.json`
  SHA256: `a782486c531f99a00247702e6aae8ffc74c7239cc64f776e1fd7679c41962301`

Bu hazırlık CPU validation: A0/3600, B0/3600; limit dışı1397. Local geçersiz70/1800; wide geçersiz1327/1800.

## Performans ve kullanım sınırları

Altı modelin her birinde CPU validation A/B0/3600. Bu paket doğrudan IK
ürünü olarak kabul edilmedi. FK_TANH limit içinde kalır fakat hedefe yeterince
yakın değildir. LOCAL_RAW local continuous hatada daha iyi olsa da wide
ve limit dışı başarısızlıkları vardır. Model aileleri farklı veri, loss,
kapasite ve eğitim bütçesiyle üretildi; aralarındaki fark tek değişkenli
nedensel deney olarak sunulmaz.

Gelecekte neural çıktı yalnız adaydır; sonlu/boyut/limit ve FK testleri,
etkin collision ve deadline koşulları ayrıca doğrulanır. Solver başarısızlığı
hedefin erişilemez olduğuna kanıt değildir. Hybrid başarısı NOT_MEASURED;
CPU batch1 latency NOT_MEASURED; eğitimden hız iddiası çıkarılmaz.

## Tekrar üretim ve depolama

Hazırlıkta mevcut ortamda CPU/thread 1/batch 1024 ile 21600 validation
çıkarımı denetlendi. Nihai T-C06 için c7798ef commit'inden temiz clone ve
sıfırdan kilitli Pixi + venv kuruldu. Python 3.12.14 / NumPy 2.5.3 /
Torch 2.10.0+cu128 / Windows 11; CPU/thread 1/batch 48. Altı checkpoint
weights_only yüklendi; 48 sabit sorguda 288 q ve bağımsız FK metrikleri
birebir eşleşti. 127 regresyon testi PASS, skip/fail/error sıfır.
CUDA yalnız decoder regresyonunda; GPU inference exact eşliği ölçülmedi.
Yeni eğitim ve Linux T-C06 çalıştırılmadı.

Çalıştırma: scripts/reproduce_c107.py; [komut ve loglar](closure/clean/),
[portable örnek](closure/witness.json), [protokol](closure/PROTOCOL.md).
Giriş JSON alanları position_m[3], quaternion_wxyz[4], q_current_rad[6].
Çıkış q_rad[6], valid, profile_a, profile_b, position_m, orientation_deg.
Geçersiz çıktı başarı sayılmaz; hata alanları null olur. Başlangıç pose'u
ve target fiziksel güvenlik veya erişilebilirlik garantisi taşımaz.

Ağırlıklar ve satır düzeyi raw LOCAL_ONLY; uzak arşiv NOT_CONFIRMED.
GitHub branch push'u bu dosyaları yüklemedi. Checkpointler optimizer dahil
tarihsel dosyalardır; yeni public inference paketi veya indirme URL'si yok.
Manifest SHA olmadan başka dosya aynı model diye yüklenmez. Silinen yerel
checkpointin uzaktan kurtarılabileceği varsayılmaz. Akademik paylaşım için
ayrıca kalıcı artifact arşivleme ve lisans/kaynak kayıtları tamamlanmalıdır.

[Manifest](preparation/handoff-manifest.json), [hazırlık denetimi](preparation/preparation-audit.json),
[nihai G1 kararı](G1_DECISION.md), [Hybrid devri](HYBRID_HANDOFF.json).

Yerel arşiv ve geri yükleme: [ara verme paketi](../../docs/records/CORE_RESUME.md).
