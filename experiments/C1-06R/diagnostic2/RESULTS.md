# Tanı 2 ve kapsamlı Core zinciri incelemesi

10 Ekim 2026 · Belge r1 · **Tanılar tamamlandı; validation hedefi NOT_MET.**

## Sekiz eşli koşunun sonucu

Tek seed 2026100901; her hücre 5000 full-batch AdamW güncellemesi, aynı
13-256-256-256-6 SiLU kapasitesi ve başlangıç parametreleri. Absolute
başlığı 0.5+f(x), residual başlığı q_current_normalized+f(x). Başlangıç
fonksiyonları farklıdır; son katman ikisinde de sıfırdır. Local/mixed
hücreler aynı hedef köklerini kullanır. Mixed, köklerin yarısında local
yerine etiketli wide satırı kullanır. Bu küçük tanı teacher başarısına
koşullu örneklem içerir; üretim başarısı veya genel veri karşılaştırması değildir.

| Örnek | Veri | Çıktı | Train Profil A | Ortak local witness A | Validation A |
|---|---|---|---|---|---|
| 64 | local | absolute | 64/64 | 32/32 | 0/3600 |
| 64 | local | residual | 64/64 | 32/32 | 0/3600 |
| 64 | mixed | absolute | 64/64 | 32/32 | 0/3600 |
| 64 | mixed | residual | 64/64 | 32/32 | 0/3600 |
| 512 | local | absolute | 0/512 | 0/256 | 0/3600 |
| 512 | local | residual | 8/512 | 5/256 | 0/3600 |
| 512 | mixed | absolute | 487/512 | 240/256 | 0/3600 |
| 512 | mixed | residual | 419/512 | 199/256 | 0/3600 |

Son adım raporlandı; daha iyi görünen ara adım seçilmedi. Tüm family/mode
ve eksik teacher satırları validation paydasında kaldı. Local-only
modelin wide başarısı sessizce dışlanmadı. Bazı mixed modellerde limit
ihlali %50'yi geçtiği için tam paydalı medyan hata sonsuzdur ve JSON'da
null olarak saklanır; yalnız geçerli satırlarla iyileştirilmiş medyan verilmez.

Residual çıktı bu tanıda çözüm olmadı. 512-local validation'da absolute
58,12 mm/9,47°, residual 78,73 mm/12,34° local medyan hataya sahip.
Q_current değiştirilmeden kullanıldığında aynı local validation'daki
47,44 mm/7,16° baseline'ın gerisindeler. Küçük train'i kusursuz ezberlemek,
gerekli lokal düzeltmeyi yeni hedeflerde öğrenmiş olmak anlamına gelmiyor.

## Optimizasyon yeterliliği kontrolü

Sekiz sonuç görüldükten sonra ayrı config kaydıyla dört 512 checkpointi
aynı veri/model/Q hedefinde LBFGS ve pozitif 1e6 kayıp ölçeğiyle en çok
500 iterasyon daha incelendi. Amaç terminal hassasiyet engelini sınamak;
optimizer ve ölçeğin etkileri ayrı ayrı izole edilmiş değildir.

| Hücre | Train A önce → sonra | LBFGS iterasyonu | Validation A sonra |
|---|---|---|---|
| local absolute | 0 → 0 /512 | 500 | 0/3600 |
| local residual | 8 → 24 /512 | 500 | 0/3600 |
| mixed absolute | 487 → 508 /512 | 500 | 0/3600 |
| mixed residual | 419 → 491 /512 | 500 | 0/3600 |

Bu bütçede terminal optimizasyonu genellemeyi kurtarmadı. Local hücrelerin
başarısızlığı yalnız erken gradient toleransıyla açıklanamaz; yine de bu
sonuç matematiksel kapasite yetersizliği veya daha fazla adımın imkânsızlığı
kanıtı değildir. Kısıtlı bütçeli tek-seed tanıdır.

## Gerçek yazılım kusurları

**1. Sınırdaki eklem dönüşümü: bulundu, yeni sürümde düzeltildi.**

Doğru normalized teacher çıktısı eski float32 `lower + z*span` yolundan
geçtiğinde, 15.204 train satırında 100 ve 3249 validation satırında 19 q
vektörü katı float64 eklem sınırının dışına taşıyor. Hata yaklaşık 1e-6
rad ölçeğinde; bu satırlar eski metrikte geçersiz sayılıyor. Teacher'ın
orijinal fiziksel q'ları ise tamamında A/B geçiyor. Sadece dtype'ı
float64 yapmak train'de üç taşmayı bırakıyor; sınır hesaplama formülü de önemli.

Yeni `c106r_precision.decode_joints` float64 ve en yakın endpointten
affine hesap kullanıyor. Eşik artırmıyor, clipping yapmıyor, gerçek limit
dışı tahmini gizlemiyor. CPU/GPU endpoint, gradyan ve geçersiz değer
testleri geçti. Yeni dönüşümle teacher train **15.204/15.204 A/B**,
validation **3249/3249 A/B**, limit ihlali sıfır.

Etki kontrolünde mevcut round1'in 12 seçilmiş modeli ve tanı2'nin sekiz
modeli aynı ağırlıklarla yeni decoder üzerinden tekrar ölçüldü. Her birinde
validation A yine 0/3600; tanı2 train başarı sayıları da değişmedi. Dolayısıyla
kusur gerçektir ama mevcut sıfır başarının temel açıklaması değildir.
Eski metrik/kod/sonuçlar korunur; düzeltme ayrı sürümlü modüldür ve sonraki
eğitim paketine yeni hash/protokolle alınacaktır. [ADR-016](../../../docs/adr/ADR-016-c106r-joint-decoding-precision.md).

**2. TorchVersion checkpoint metadata: önceden bulunan kusur doğrulandı.**

Round1 production metadata'sı weights-only okuyucuda özel sınıf gerektiriyor.
Yeni tanı checkpointleri `str(torch.__version__)` yazar; sekiz dosya
allowlist olmadan güvenli biçimde yüklenip train çıkarımı birebir doğrulandı.
Eski dondurulmuş launcher'ın resume kusuru tarihsel olarak açık kalır;
sonraki uzun eğitim paketi gerçek production contract ile resume testini
geçmeden teslim edilmeyecek. Bu metadata sorunu öğrenilmiş ağırlıkların
geometrik doğruluğunu açıklamaz.

## Kapsamlı incelemede doğrulananlar

| Katman | Gerçekte yapılan kontrol | Sonuç |
|---|---|---|
| Robot/TCP/birim/joint sırası | G0 byte kilidi; URDF/spec ve robot testleri | PASS |
| FK ve türev | 1086 fixture × CPU/GPU × float32/64; 32 gradient fixture/cihaz; physics/tanh kontrolleri | PASS |
| Bağımsız FK | Tüm 18.453 etiketli train/validation q üzerinde Pinocchio–IndependentFK | 1e-9 m/matris eşiği PASS |
| Veri kökeni | 20.400 satırda q_current seed/üretim tekrarı, hedef-root FK eşliği | PASS |
| Teacher ve normalizasyon | 18.453 etiket A/B; train-only mean/std; sıra/ölçek/NaN negatifleri | PASS |
| Split | Train/validation source-root ve grup ayrımı; sealed split reddi | PASS |
| Model/başlık/kayıp | Eşli parametreler, çıkarım geri yükleme, ham limit ihlalleri, gradyan kontrolleri | PASS; düşük pose başarısı ayrıca raporlandı |
| Metrik/payda | Yanlış/eksik/limit dışı satırları paydada tutma; oracle kontrolü | Endpoint kusuru ayrıldı ve yeni sürüm doğrulandı |
| Checkpoint/tamlık | Önceki round1 85 SHA ve 24.000 epoch denetimi korundu; yeni raw/weight SHA | PASS |

Regresyon kapsamı: F0-01 **16**, F0-02 **102**, F0-03 **159**, C1-03
**110**, C1-04 **12**, C1-05 **36**, C1-06R **33** ve yeni precision
testleri **6**: **474 test PASS, 0 fail/skip**. C1-03 NaN mutantının
determinant hesabında üretilen bir uyarı ham kayıtta saklıdır. Tanı2
başlangıçtaki üç test 33 içinde tekrar yer aldığı
için toplama ikinci kez eklenmedi.

Bu sayı bütün deponun bütün koşullarının yeniden çalıştırıldığı iddiası
değildir. C1-02'nin tüm test/benchmark ham verisini okuyan tam kabul koşusu,
eski final değerlendirme ve Docker/harici solver kampanyası yeniden
çalıştırılmadı. C1-02 train/validation zinciri doğrudan yeniden denetlendi.
Yeni final üretilmedi; eski final ham sorguları bu tanılarda açılmadı.

## Nerede yanlış yola gittik?

**Kanıtlanan süreç sorunu:** 64 örnekte overfit başarısını uzun eğitime
hazırlık için fazla güçlü yorumladık. Yeni ölçek tanısında aynı yöntem
512 local satırın çoğunda bile pose hassasiyetini sağlayamıyor. Uzun
eğitime geçişte yalnız yazılım testleri ve 64 örnek kapısı yeterli değil;
ölçek arttıkça train hassasiyeti ve validation eğrisi ayrıca izlenmeli.

**Kanıtlanan model davranışı:** Mevcut mutlak pose + q_current girdileri
ve tek eklem vektörü regresyonu, bu eğitim rejimlerinde küçük örnekleri
ezberliyor; yeni sorgularda doğru yerel düzeltme öğrenemiyor. Residual
offset eklemek tek başına bunu düzeltmedi. Bu davranış kesin tek bir
mimari kusuru kanıtlamaz; veri/temsil/optimizasyon birlikte sınanmalı.

**Ölçülmüş veri zorluğu:** Aynı hedef pose için local/wide teacher q
vektörleri belirgin farklı olabiliyor. Eşli etiketlerde joint L2 farkının
medyanı train 5,815 rad, validation 5,901 rad. Bu iki geçerli q'nun
aritmetik orta noktası train 6804 hedefin yalnız 743'ünde, validation
1449 hedefin 163'ünde A geçiyor. Ancak q_current girdileri farklı:
bu ölçüm aynı girdiye çelişkili label hatası veya modelin gerçekten
dal ortalaması aldığına dair nedensel kanıt değildir. Tek-vektör MSE
yaklaşımının çözüm geometrisindeki zorluğunu gösteren kontrollü tanıdır.

**Eksik teacher etkisi:** Train'de 1596, validation'da 351 wide label
yok. Supervised train satırlarının yaklaşık %55,25'i local; değerlendirme
%50 local/%50 wide. Etiketli filtrelemenin bu etkisi kayıtlı ve bütün
eşli arm'larda ortak; eksik etiketleri sıfır cevap gibi öğrenme hatası
bulunmadı. Veri kapsamı genişletilirse ayrı sürüm ve eşli kontrol gerekir.

## Sonraki kontrollü yön

Yeni uzun eğitimi tekrar başlatmıyoruz. Bir sonraki temsil tanısında aynı
13 giriş/aynı kapasite korunarak mutlak hedef pose yerine, q_current'ın
FK'sine göre **konum farkı ve göreli yönelim** verilmesi sınanmalı. Bu,
mevcut 13 girdiden deterministik ön işlemdir; sayısal IK düzeltme adımı
veya Hybrid çözümü değildir. Ham/göreli pose × absolute/residual dört
hücresi aynı köklerde karşılaştırılmalı; normalizasyon train-only, decode
ADR-016 olmalı. Önce sayısal config ve ADR kaydı, sonra kısa tanı.

Bu öneri henüz uygulanmış veya başarılı değildir. Önce 512 ve daha geniş
train ölçeğinde pose hassasiyetinin, ayrı local/wide validation eğrilerinin
iyileştiği gösterilmeli; ardından üç-seed uzun paket tasarlanır. Feature
dönüşümü de işe yaramazsa kapsamlı veri/çözüm dalı modelleme deneyi gerekir.
%95 hedefi, A/B eşikleri ve final sınırı değişmez. C1-06R açık; C1-07/G1
başlamadı. Şu anda kullanıcıdan yeni uzun eğitim çalıştırması gerekmiyor.

Makine kanıtları: [results.json](results.json), [refinement](refinement/results.json),
[pipeline](project-review/pipeline.json), [düzeltmenin etkisi](project-review/precision-fix.json),
[çalışma kaydı](RUN_REPORT.md).
