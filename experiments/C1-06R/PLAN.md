# C1-06R — Core başarı iyileştirme planı

9 Ekim 2026 · Belge r1 · Yazılım hedefi v1.0.0

Durum: **PLAN_PROPOSED / IMPLEMENTATION NOT_STARTED / TRAINING NOT_RUN / NEW_FINAL_TEST NOT_CREATED**.

Bu belge, kullanıcının paylaştığı `C1-06R_Basari_Iyilestirme_ve_Yeni_Bagimsiz_Test_Promptu.md` metninin depo kanıtlarıyla değerlendirilmesi ve birlikte çalışma planıdır. Ekli belgedeki yürütme/commit/push komutları bu inceleme isteğiyle otomatik uygulanmış değildir. C1-07 görevdir; Core içindeki kapanıştır, ayrı faz değildir. Önerilen sıra C1-06 (korunan sonuç) → C1-06R → C1-07 → G1 kararıdır. Hybrid fazı başlatılmaz.

## 1. Mevcut durum ve hedef farkı

| Başlık | Mevcut kanıt | İyileştirme hedefi |
|---|---|---|
| Robot/FK/gradyan | C1-03 T-C01/T-C02 PASS / ACCEPTED | G0 sözleşmesini koru; yeni eğitim yolunu ayrıca doğrula |
| Doğrudan neural IK | C1-06: 21 checkpointin her birinde A/B 0/12.000 | Yeni bağımsız testte Profil A ≥%95 ürün hedefi |
| FK/limit katkısı | Eski H2 REJECTED; main ve zor fark 0 yp, ampirik CI [0,0] | Yeni H2-R: zor kümede ≥+2 yp, main ≥−1 yp |
| Limit uyumu | FK_TANH üç seedde sıfır limit ihlali; yine sıfır başarı | Limit uyumu ve hedef poza ulaşma birlikte |
| Validation | FK_TANH: medyan 137–283 mm, 25–62°; A/B 0/3.600 | Önce train/validation üzerinde mekanizmayı ayır |
| Core kapanışı | T-C05 PASS; C1-07/T-C06 ve G1 açık | Model kartı, temiz tekrar ve dürüst faz devri |

Kaynaklar: [Core raporu](../../docs/raporlar/02_Core_v1_0_r1.md), [C1-05 validation](../C1-05/stage2/RESULTS.md), [C1-06 sonuçları](../C1-06/stage2/RESULTS.md), [C1-06 kabulü](../C1-06/stage2/final-001/acceptance.json).

Core raporu §6, %95'i ürün hedefi olarak tanımlar; araştırma kapanışını olumlu H2 sonucuna bağlamaz. Kullanıcının amacı bu ayrımdan yararlanarak erken kapanmak değil, C1-07 öncesinde başarıyı gerçekten iyileştirmektir. %95'e ulaşmak şu aşamada garanti edilemez.

## 2. Prompt değerlendirmesi ve gerekli netleştirmeler

Korunacak güçlü yönler: eski negatif sonucun değişmezliği; ürün ve hipotez kararının ayrılması; final testten bağımsız teşhis; eşli kontroller; üç seed; bağımsız FK; başarısız satırları paydada tutma; AI/kullanıcı iş bölümü.

Uygulamadan önce şu noktalar protokole açıkça yazılmalı:

1. **Ürün paydası ve seed kuralı:** Öneri: 10.000 main sorguda, üç ön kayıtlı seedin her biri için A ≥%95; local/wide ve iki zor küme ayrıca raporlanır. 12.000 tüm sorgu oranı da verilir. Bu, kaynak raporda tanımlanmamış payda/seed ayrıntısı için yeni protokol önerisidir, eski kabul şartı değildir. Main başarısı tüm alt kümelerde %95 diye sunulmaz. Ürün kararı örneklem oranına dayanır; popülasyon için %95 alt güven sınırı iddiası ayrı ve daha güçlüdür. Eksik/bozuk kampanya INCONCLUSIVE olur.
2. **H2-R karar kuralı:** Mevcut `c106.py:h2_decision` ile tutarlı öneri: root bootstrap %95 CI alt sınırı zor kümede ≥+2 yp ve main'de ≥−1 yp ise SUPPORTED; CI üst sınırı bu eşiklerden herhangi birinin altındaysa REJECTED; aksi INCONCLUSIVE. Nokta tahmini ayrıca raporlanır. Teknik bütünlük yoksa teknik nedenle INCONCLUSIVE. Bu kural finalden önce dondurulur.
3. **Tavan etkisi:** Q kontrolü zor kümede %98'in üstüne çıkarsa +2 yp artış matematiksel olarak mümkün olmaz. Kontrol zayıflatılmaz, eşik düşürülmez. İyi ürün sonucu ile reddedilmiş H2-R birlikte mümkün olabilir; kullanıcı hedeflerinin ikisinin de mutlaka sağlanacağı taahhüt edilmez.
4. **Yönelim kaybı:** Depoda eğitim chordal, raporlama geodezik açıdır. Promptun “geodezik rotasyon kaybı” ifadesi otomatik kayıp değişikliği sayılmamalı. Önce mevcut kayıp/gradyan doğrulanır; eğitim kaybı değişirse ayrı ablasyon ve ADR gerekir.
5. **Adil eğitim:** Aynı veri/kapasite/seed ve aynı optimizer güncelleme bütçesi sabitlenir; FK eklemenin gerçek duvar süresi ayrıca ölçülür. Eşit adım ile eşit duvar süresi aynı şey değildir. Kontroller de aynı seçim ve arama fırsatını alır.
6. **Yeni baseline:** Eski 600.000 baseline satırı yeni sorgulara eşlenemez. Yeni final sorgularında solverlar yeniden çalıştırılmalıdır. Farklı platformlar geometri bakımından bağımsız doğrulanabilir; hız üstünlüğü için aynı host/runtime/kaynak ve ölçüm sınırı gerekir.
7. **Mühür:** SHA değişmezliği gösterir, erişimi tek başına engellemez. Ayrı evaluator erişimi, eğitim paketinden finalin çıkarılması ve erişim/açılış kaydı gerekir. Henüz üretilmemiş sete SEALED yazılmaz.
8. **Süre ve tur sayısı:** Kullanıcı süre sınırı olmadığını belirtti. İlk iki tur bir başlangıç planıdır, nihai araştırma sınırı değildir. Her turun konfigürasyon ve deneme sayısı yine önceden sınırlanır; sonraki tur yalnız validation bulgusuyla yeni revizyonda tanımlanır. Açılmış final yeniden arama aracı yapılmaz.
9. **Hybrid anlamı:** Core raporu, düşük doğrudan IK başarısı olan geçerli bir modeli araştırma amaçlı H1 başlangıç adayı olarak taşıyabilir. Doğrudan IK NO-GO ile Hybrid araştırma adaylığı ayrıdır; Hybrid ürün hazır oluşu ölçülmeden iddia edilmez.

## 3. İlk teşhis öncelikleri

| Öncelik | Mevcut bulgu | Sınanacak açıklama ve ayırıcı kontrol |
|---|---|---|
| P0 — eğitim/değerlendirme zinciri | Önceki FK ve veri kabulü var; yeni GPU yolu yok | Target↔teacher FK, frame/TCP, rad/m, joint sırası, normalize→fiziksel dönüşüm, train/inference eşliği; yanlış sıra/etiket negatif kontrolleri |
| P1 — seçim ve durdurma | `c105.py` checkpointi validation Lq ile seçiyor; E-C05 seed1/3 best epoch 7/8, toplam 27/28 | Train/validation üzerinde Lq, pose hatası, A/B ve best/last ilişkisi; yeni koşuda geometriye göre seçim ablasyonu. Bu bir aday mekanizma, kanıtlanmış yazılım hatası değil |
| P2 — öğrenme yeterliliği | 15.204 etiketli train; 256×3 MLP; eski üst sınır 3.000 optimizer adımı | Küçük tutarlı sette gerçek Profil A öğrenmesi, öğrenme eğrileri ve veri ölçek eğrisi; az eğitim/kapasite/etiket çatışmasını ayır |
| P3 — çoklu çözüm ve öğretmen | Pose-only aynı hedefte farklı q etiketleri; train'de 1.596 eksik wide etiket | Conditioned girdinin yakın komşularında dal tutarlılığı, teacher başarı/başarısızlık ve dağılım kapsamı; pose-only bulgusunu conditioned modelin nedeni sayma |
| P4 — kayıp ve bounded head | FK yönelimi iyileştiriyor; tanh limit ihlalini bitiriyor ama başarı getirmiyor | Lq/Lp/LR ölçekleri, bileşen gradyan norm/yönleri, tanh saturasyonu ve sınır davranışı; kayıp ve başlığı ayrı tut |

Ek gözlem: C1-04 seed1 epoch200 Lq train/validation 0,1513/0,1711; wide 0,3157/0,3591, local 0,0182/0,0198. Bu kayıt tek başına underfit veya dal ortalaması teşhisi değildir. Ayrıntılı nedenler yeni tanılarla doğrulanacaktır. Eski finalin hata satırları eğitim örneği veya model seçimi girdisi yapılmayacaktır.

## 4. Birlikte uygulama sırası ve geçiş kapıları

| İş paketi | AI'nın işi ve teslimi | Kullanıcının işi | Bir sonraki adıma geçiş |
|---|---|---|---|
| R0 — protokol/ortam | C1-06R görev kaydı ve ADR; input byte/SHA; ayrı CUDA ortamı; runtime, kaynak ve smoke denetimi | Gerekiyorsa makinedeki erişim/kurulum engelini bildirme | G0 değişmez, runtime smoke ve bağımsız FK/gradyan eşliği PASS |
| R1 — kök neden | Train/validation ve sentetik tanılar; veri/etiket/soy/normalizasyon denetimi; küçük overfit; bulgu→kontrol→sonuç raporu | Uzun eğitim yok | Açık eşleme/gradyan hatası yok; küçük sette gerçek pose başarısı gösterilmiş; açıklanamayan başarısızlık varsa uzun eğitim durur |
| R2 — veri ve seçim | Gerekirse sürümlü yeni train/validation; kök aile ayrımı; train-only normalizasyon; selection/curriculum/architecture gerekçesi | Uzun eğitim yok | Yeni veri audit PASS; kontrol/candidate aynı veri ve seçim bütçesi; protokol dondurulmuş |
| R3 — eğitim paketi | Çalıştırılmış launcher, konfigürasyon, resume, kaynak pilotu, log/checkpoint denetleyici; makineye özgü tek komutluk yönerge | Hazır uzun eğitim komutunu çalıştırma | READY_FOR_USER_TRAINING; gerçek koşu başlamadan TRAINING NOT_RUN |
| R4 — validation turu | Çıktı/hash/tamlık denetimi; üç seedli kıyas; başarısız koşu tanısı; validation temelli sonraki tur kararı | Hazır düzeltilmiş/ikinci tur komutunu çalıştırma | Ön kayıtlı aday seçimi; seed eleme yok; hedefi desteklemeyen validation'da finali açma |
| R5 — bağımsız final | Yeni sorgu/solver/checkpoint/code hash freeze; bağımsız FK; root bootstrap ve hata raporu | Yalnız erişim gerekirse hazır final komutunu çalıştırma | Açılış kaydı sonrası tek final kampanyası; MET/NOT_MET/INCONCLUSIVE ile H2-R ayrı |
| R6 — C1-07 | Eski ve yeni sonucu birlikte model kartına/devir manifestine işleme; temiz çıkarım/değerlendirme; G1 değerlendirmesi | Modelin amaçlanan kullanımını değerlendirme | T-C06 ve bütün G1 koşulları tamamlanmadan sonraki faz yok |

R1 küçük overfit için önerilen tanı: önceden sabitlenmiş 64 tutarlı train witness üzerinde A 64/64, sınır/singularity fixture ve negatif kontroller ayrıca. Bu yeni bir **tanı kapısı önerisidir**; kaynak T-C03 eşiğini değiştirmez ve genelleme kanıtı değildir. Epoch/adım bütçesi tanı çalışmadan önce dondurulur. Geçmezse eşik indirmek yerine neden araştırılır.

## 5. Deney matrisi ve model seçimi

İlk hedef güçlü ve doğru bir conditioned Q kontrolü kurmaktır. R1 bulgusu olmadan daha büyük model, 6D, delta-q ve çoklu aday aynı anda eklenmez.

Başlangıç matrisi önerisi:

| Arm | Eğitim kaybı | Çıkış | Amaç |
|---|---|---|---|
| Q | supervised Q | ortak seçilmiş başlık | Eşli ana kontrol |
| Q+FK | Q + FK konum/yönelim | Q ile aynı | FK katkısını ayır |
| Q+FK+limit | Q + FK + limit cezası | aynı unbounded başlık | Yalnız limit ihlali bulgusu gerekçelendirirse |

Bounded head asıl aday olacaksa alternatif minimal 2×2 matris: Q/unbounded, Q/bounded, Q+FK/unbounded, Q+FK/bounded. Birincil H2-R kontrastı, salt FK mı yoksa FK+başlık paketi mi ölçtüğünü açıkça tanımlar. Limit cezası ile bounded head gereksiz yere birlikte eklenmez. Her tam arm üç seed; 2–4 arm ile tur başına 6–12 model-seed koşusu. Gerçek arm sayısı R1 sonunda sabitlenir.

Yeni koşullarda checkpoint seçimi için öneri: validation Profil A oranı, eşitlikte daha düşük geçersiz oranı, ardından tam paydalı normalize pose hata metriği; tam eşitlikte erken epoch. Başarı başlangıçta sıfırsa sürekli pose metriği öğrenmeyi izleyebilir. Pose metriğinin formülü, geçersiz değer kuralı, sıralama, ölçüm sıklığı ve early stopping uygulama protokolünde **koşmadan önce** sayısallaştırılır. Geodezik açı raporlama ile differentiable eğitim loss'u karıştırılmaz. Eski checkpointler bu yeni kuralla yeniden C1-06 sonucu olarak seçilmez.

İlk tam turda eşli arm'lara sabit ve aynı güncelleme sayısı önerilir; q-loss patience ile bir arm'ın diğerini erken durdurması önlenir. Öğrenme oranı planı ve gradient clipping ihtiyacı küçük pilotla belirlenir. İkinci tur ancak birinci turun train/validation bulgusuyla gerekçelendirilir; tüm başarısız denemeler kayıtlı kalır. Veri miktarı, batch, epoch ve λ değerleri bu inceleme sırasında kanıtsız seçilmemiştir.

Validation final açılışına hazır değilse eğitim/tanı döngüsü sürer. Main'de önerilen üç-seed %95 validation kapısı final başarısını garanti etmez. Final açıldıktan sonra yapılacak model değişikliği yeni deney kimliği ve yeni bağımsız test gerektirir.

## 6. Veri ve final test sözleşmesi

- Eski C1-02, Foundations ve C1-06 girdileri sabit kalır. Yeni dataset sürümü, üretim seed'i, root soy ağacı, şema ve hash taşır; train/validation/final kök aileleri ayrıdır. Yalnız farklı random seed kullanmak bağımsızlık kanıtı değildir; exact/near duplicate ve aile örtüşmesi denetlenir.
- Teacher başarılı/eksik, family ve local/wide sayıları split başına verilir. Eğitim etiketleri bağımsız FK ile doğrulanır. Etiketsiz satırların FK eğitimine alınması Q karşılaştırmasının veri kapsamını değiştirir; ana karşılaştırmada sessiz ekleme yapılmaz, gerekiyorsa ayrı yarı denetimli deney açılır.
- Yeni final: 10.000 main = 5.000 local + 5.000 wide; en az 1.000 boundary ve 1.000 singularity. Kümeler ayrık, root ailesi kümelendirmesi kayıtlı. Local yarıçapı ve q_current üretimi sonuçtan önce sabitlenir; eğitimdeki ±0,1 ile eski benchmark ±0,05 rad farkı açıkça ele alınır.
- A: ≤0,002 m ve ≤1°; B: ≤0,001 m ve ≤0,5°; sonlu/doğru boyut/limit içi q zorunlu. Tam paydadan başarısız satır çıkarılmaz. Direkt model çıktısına sayısal refinement eklenirse sonuç doğrudan neural başarıya yazılmaz.
- Yeni finalin dağılımı erken dondurulur. Sorgular eğitimden ayrılmış evaluator sürecinde üretilir ve kalite denetiminden geçer; yalnız integrity özeti görünür olur. Ham sorgular ve sonuçlar eğitim ortamına bağlanmaz. Tam mühür, dosya SHA ve erişim günlüğü tamamlanınca SEALED durumuna geçilir.
- Yeni sorgularda DLS/KDL/TRAC-IK/pick_ik local/global ölçümleri yeniden yapılır; mevcut baseline ayarları korunur veya değişiklik validation'da ön kaydedilir. Hız kıyası için neural ve baseline aynı platform/kaynak sözleşmesinde çalıştırılır. 10/50 ms deadline ile A/B geometri ayrı raporlanır.
- Eşli query_id/root/seed; root düzeyinde, alt küme yapısını koruyan 10.000 bootstrap tekrarı ve ön kayıtlı RNG seed'i. Üç seed eşit ağırlıklı; her seed ve aralarındaki değişkenlik ayrıca. Beş latency geçişi bağımsız sorgu sayılmaz. Eşleşme eksikliği önce tamlık sorunu olarak durdurur; sessiz inner join yok. Geçerli bir model başarısızlığı tam paydada failure'dır.
- Raw q, hata sınıfı, A/B, local/wide/boundary/singularity, medyan/P95/P99 pose hatası, limit oranı ve maliyet saklanır. Geçerli-only dağılımın coverage'ı verilir; invalid'ler için tam paydalı açık politika uygulanır.
- Final sonucu başarısızsa hedefler gevşetilmez. C1-07 eski H2 reddini ve yeni sonucu birlikte taşır. Collision/fiziksel robot NOT_CHECKED kalır.

## 7. Donanım ve eğitim teslimi

Kullanıcı: RTX 5060, 24 GB RAM; süre sınırı yok; kişisel bilgisayarda local veya Docker kullanılabilir.

Yerel `nvidia-smi` gözlemi: **NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB VRAM, sürücü 596.21**. Bu gözlem CUDA/PyTorch eğitim uyumluluğu veya Docker GPU erişimi testi değildir. Geçmiş Core ortamı CPU Torch overlay kullanır.

R0'da ayrı ortam kurulur; Foundations kilidi değiştirilmez. Önce native Windows GPU smoke, gerekli olması halinde WSL2/Docker seçeneği değerlendirilir; tek canonical eğitim yolu belirlenir. Kurulum sürümleri resmi uyumluluk bilgisi ve gerçek forward/backward/FK smoke ile sabitlenir. Sadece `cuda.is_available()` yeterli kabul edilmez.

VRAM/RAM/boş disk, epoch süresi ve checkpoint maliyeti gerçek pilotla ölçülür. Süre tahmini = ölçülen epoch/step maliyeti × kayıtlı bütçe + validation/checkpoint payı; henüz süre veya batch sığma garantisi yoktur. İlk doğrulama float32; AMP gibi sayısal değişiklikler ayrı eşlik kanıtı gerektirir. Bellek gerekirse microbatch/gradient accumulation ile yönetilir; tam epoch kapsamı korunur.

Teslim paketi: exact code SHA; kurulum ve hash kontrolü; sıralı launcher; atomik best/last checkpoint; model/optimizer/scheduler/RNG/sampler/global-step/config kimlikleri; kesinti sonrası eşdeğer resume kontrolü; JSONL train/validation kaynak logları; completion manifest ve SHA256SUMS. Başarı/hatada paylaşılacak dosyalar otomatik paketlenir. Kullanıcı YAML, komut veya veri düzenlemek zorunda kalmaz; hata analizi AI'dadır. Ağırlık/veri normal Git'e eklenmez.

## 8. İzlenebilirlik ve mevcut teslim

| Gereksinim | Planlanan değişiklik | Planlanan test | Kanıt |
|---|---|---|---|
| REQ-C02 | Yeni runtime ve eğitim FK yolu | CPU/GPU/FK/FD/inference eşliği | R0/R1 runtime ve tanı raporları |
| REQ-C03 | Sürümlü veri, doğru hedef ve model seçimi | Split/teacher/overfit/negatif kontroller | Veri manifesti, tanı raporu, config |
| REQ-C04 | Eşli kontrollü yeni deney | Üç seed, aynı bütçe, loss/head ablasyonu | Train logları, checkpoint manifesti |
| REQ-C05 | Yeni bağımsız final ve yeni baseline | Seal, FK, join, bootstrap/tamlık | Final raw, CI, iki ayrı karar |
| REQ-C06 | C1-07 devir/model kartı | Temiz ortamda örnek çıkarım/değerlendirme | MODEL_CARD, G1, handoff |

Bu tur yalnız inceleme ve plan üretir. Yeni runtime, tanı testi, veri üretimi, eğitim, final ve T-C06 **NOT_RUN**. Yeni ADR/görev/protokol henüz dondurulmadı; READY_FOR_USER_TRAINING veya SEALED iddiası yoktur. Mevcut STATUS ve TRACEABILITY kullanıcı değişiklikleri içerdiğinden bu plan turunda düzenlenmedi; R0'da yalnız yeni görev ekiyle birlikte güncellenecek.

Bir sonraki somut iş: **R0/R1 — ayrı GPU ortamı ve train/validation kök neden tanı paketi**. Önce doğru öğrenmenin küçük ölçekte kanıtı, sonra uzun eğitim teslimi.
