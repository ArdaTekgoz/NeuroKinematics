# C1-05 test ve negatif kontrol matrisi

8 Ekim 2026 · r1. Mevcut kapılar ve onay sonrası uygulanacak kapılar ayrıdır.
T-C04 **NOT_RUN**; hiçbir tasarım/manifest denetimi T-C04 yerine geçmez.

| Kapı | Girdi / sabit kabul | Kanıt / bugünkü durum |
|---|---|---|
| Girdi kimliği | C1-04 frozen 62 girdi/12 çıktı, 34 shard; altı checkpoint byte/SHA; kabul/normalizasyon | `commands/inherited-stage1`, `input-access.json`: PASS |
| Tarihsel çıkarım | 6 × 3.600 validation q, özgün batch1024; fark tam sıfır | `commands/input-access-v2`: PASS; eski Profile A 0/3600 korunur |
| Regresyon | C1-04 12, C1-02 7, C1-03 110; public limit reddi ve 24 gerçek source mutant dahil | `regression/junit.xml`: 129 PASS, skip0 |
| T-C01/T-C02 tam regresyon | Frozen 1086 q/dtype, 32 gradient/gradcheck/Jacobian, batch/edge/sensitivity | `regression/fk-full`: PASS; eski eşikler değişmedi |
| Yeni FK ileri alanı | `fk-domain-samples.json`: 32 iç +48 kontrollü dış q; float64 konum/R ≤1e-9; float32 ≤1e-5 | NOT_RUN; Stage2 zorunlu |
| Yeni FK backward | 80 örneğin tümünde 12 pose çıktısı ve Lq/Lp/LR/toplam için h=1e-6 merkezi fark; abs ≤1e-5+1e-3×abs(FD); Torch gradcheck aynı tolerans | NOT_RUN; Pinocchio FD + analitik türev, valid mask yok |
| Analitik fixture | 1R ve 2R zincir, nonidentity base/TCP, origin-before-motion, negatif/π/π±1e-6/2π açıları; konum/R/analitik türev aynı eşikler | NOT_RUN; robot parserına bağlı olmayan kapalı form oracle |
| Domain stres | float32/64 en büyük sonlu ±değer her eklemde, batch1/2/7/1024, dtype/device, NaN/Inf/yanlış sıra/birim | NOT_RUN; aşırı açıda finite çıktı, FD doğruluğu iddiası yok |
| Pilot | 192 train/96 validation sabit ID, 20 update/arm; bileşen ve katman/çıktı gradyanları, NaN/Inf, limit dışı oran, ≥175° altkümesi | NOT_RUN; pilot ağırlıkları ana eğitime taşınmaz |
| E-C03 | Q/FK × üç seed; başlangıç state/permutation hashleri, aynı etiketli envanter/bütçe; q-loss checkpoint seçimi | NOT_RUN |
| E-C04 | FK/FK_LIMIT × üç seed; tek değişken normalize ReLU cezası; sınırsız başlık | NOT_RUN |
| E-C05 koşullu | FK/FK_TANH × üç seed; lambda_lim=0; sınır altkümesinde dq/dz, doygunluk ve limit oranı | NOT_RUN; koşul `config.json` |
| Kanıt / temiz çıkarım | Her koşu epoch kayıpları/gradyanlar, argv/exit/log, checkpoint kimlikleri, 3600 ham satır; temiz checkout yükleme/çıkarım/audit | NOT_RUN; temiz ortamda tekrar eğitim ayrı NOT_RUN olabilir |

## Zorunlu negatif ve kaynak-mutant kontrolleri (Stage2 NOT_RUN)

| Kontrol | Beklenen yakalama / neden |
|---|---|
| λp=λR=0 | Aynı başlangıçta Q ile toplam loss ve tüm parameter gradyanları eş; FK etkisi iddia edilmez |
| FK output detach / no_grad / NumPy conversion / zero-gradient backward | İzole Lp ve LR autograd/FD ve nonzero fixture testi FAIL; combined q loss'un gradyanı bu kusuru örtemez |
| Yanlış base/world veya tool0 yerine flange | Analitik nonidentity base/TCP ve Pinocchio forward karşılaştırması FAIL |
| wxyz→xyzw, yanlış quaternion işareti işleme | Aynı rotasyonu temsil eden u/-u için R ve LR eş; π çevresi finite; ingress canonicalization; yanlış sıra rotasyon oracle'ında FAIL |
| q_target'ı feature'a sızdırma | Etiketleri permüte edip feature bytes/ID aynı kalmalı; forbidden alan/feature width kontrolü FAIL |
| Sahte q_target eşlemesi | Sabit cyclic label shift, bağımsız FK Profile B eşleşme denetiminde reddedilir; eğitim başlamaz |
| Yanlış m/mm, rad/deg, ℓ, /8 veya joint reduction | Elle hesaplanan Lp/LR/Lq ve autograd/FD oracle'ı FAIL; bileşen ölçeği epoch loss düşüşünden çıkarılmaz |
| Clamp/wrap/drop-invalid ekleme | Dış q fixture'da `q_raw==q_fk_input==q_eval` ve analytic gradient FAIL; bütün satır sayısı doğrulanır |
| Etiketsiz wide satır silme | 3600 unique pair_id, 351 unlabeled ve split-order SHA kapısı FAIL |
| Yanlış limit cezası | İçte sıfır, altta/üstte normalize kare ve işaretli gradient; sınırda zero subgradient; sınır noktası iç-FD testine karıştırılmaz |
| Tanh ile limit cezasını aynı anda değiştirme | Matrix isolation check FAIL; E-C05 lambda_lim sıfır kalır |
| En iyi FK epoch'una geçiş / seed seçme | Epoch logundan en düşük finite Lq ve ilk eşit epoch tekrar seçilir; tüm üç seed zorunlu |
| Test/benchmark loader açma | Model seçim API'si train/validation dışında exception verir; dosya erişim spy testi test shardını açmayı yakalar |
| Checkpoint/config/SHA veya command log bozma | Audit fail closed; yeni manifest yazarak geçirme yok |

Negatif kontroller fixture testleriyle sınırlı kalmaz: Stage2, her ilgili üretim
kod yoluna geçici gerçek source mutant uygular, diff/killing-test/exit kaydeder.
SURVIVED veya ERROR teknik kusurdur. Q/FK ölçekleri ölçülmeden tam eğitim yok.

## Yorumlama

Chordal LR=sin²(θ/2), π çevresinde küçük gradyan gösterebilir; bu beklenen
geometri kaydedilir. Tam π noktasında sıfır türev graph kopması sayılmaz;
nonstationary fixture'lar ayrıca zorunludur. Pilot teknik geçerlilik kapısıdır,
genelleme veya Profile A iyileşmesi garantisi değildir. C1-04'ün mevcut küçük
öğrenme kontrolü yeni physics-aware pilotu yerine geçmez.
