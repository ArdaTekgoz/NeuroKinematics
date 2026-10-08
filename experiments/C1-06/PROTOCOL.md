# C1-06 bağımsız değerlendirme ön kaydı

8 Ekim 2026 · r1 · REQ-C04, REQ-C05 / T-C05 NOT_RUN.
Bu belge nihai sonuç görülmeden sabitlenmiştir. Kaynak görev, SOURCE_REQUEST.md;
otorite kullanıcı devralma isteği, depo AGENTS.md ve korunmuş C1-06 görev planıdır.
Nihai test için bu kayıt commit/push edildikten sonra ayrıca açık Aşama 2 onayı gerekir.

## Kimlik ve açılış kapısı

G0 robot/URDF/TCP, joint_1…joint_6 sırası, radyan/metre, base_link→tool0,
wxyz quaternion ve eklem limitleri input-hashes.json ile sabittir.
C1-01 udp-v2 query SHA `120b41f07109aeaca10e10fbb04783167cdc4ba282e7976941468c7dccfa4976`.
C1-02 kaynak/normalizasyon, C1-05 devir/seçim config'i ve 21 checkpoint gerçek
bayt/SHA denetiminden geçmiştir. Devirdeki FK_TANH üç seed'i korunur; tek seed seçilmez.

Aşama 1'de test/benchmark JSONL veya NPZ satırları parse edilmez; yalnız byte
hash, boyut ve newline sayısı alınır. Girdi kapısı geçmiş manifest/gate/schema
ve kaynak üretim koduna dayanır. Satır bazlı query join henüz NOT_RUN'dır.
Mevcut tarihsel kabul raporları incelenmiştir; bunlar neural final sonuç değildir.

Onay sonrası önce SHA256SUMS ve bütün input-hashes doğrulanır. 18 C1-05 best
checkpoint için fixed-validation-witness.json içindeki 10 sabit girdinin q/FK
eşliği, C1-04 üç conditioned model için mevcut C1-04 witness eşliği denetlenir.
Yeni loader ve değerlendirme yürütücüsü sentetik negatif testlerden geçer.
Yürütücü Aşama 2'de tamamlanacak uygulama işidir; burada istatistik/karar ve
girdi/bağımsız doğrulama çekirdeği dondurulmuştur. Yeni yürütücü kodunun SHA'sı
final açılmadan kaydedilir; bu protokol veya model/eşik seçimi değiştirilmez.

## Sorgu ve soy eşlemesi

Tek ana kaynak, F0-05/C1-01'in mevcut 12.000 sorguluk dosyasıdır: main 10.000
(5.000 local/5.000 wide), boundary 1.000 (500/500), singularity 1.000 (500/500).
Dosya sırası korunur. query_id=f05-{subset}-{index:08d},
query_group_id=root-f05-{subset}-{index:08d}; bunlar üretici sözleşmesidir,
bu oturumda satırdan okunmuş gözlem değildir. q_current mevcut alanından aynen
alınır; yeniden üretim, etiket kopyalama veya neural modele özel başlangıç yoktur.

F0-05 local perturbasyon aralığı ±0,05 rad; wide bağımsız uniform başlangıç,
normalize RMS uzaklık en az 0,25 filtresi. Bu tarihsel dağılım C1-02'nin
±0,1 rad local ve öğretmenli wide eğitim dağılımından farklıdır. Değiştirilmez;
dağılım kayması sonuçlarda belirtilir. Teacher başarılarıyla sorgu seçilmez.
C1-02'nin 3.600 test çifti 12.000 benchmark yerine veya paydaya eklenmez;
bu ön kayıt kapsamında C1-02 test shardları mühürlü kalır.

Onaylı açılışta model çıkarımından ÖNCE tam kimlik audit'i yapılır:

1. Manifest hash/count/50:50, unique query_id ve group, kaynak aile, q_current
   sonluluk/limit/provenance; q_target yalnız kaynak audit içindir.
2. Hedef poz bağımsız FK ile G0 toleranslarında doğrulanır. C1-02 kabulündeki
   benchmark soy ayrımı yeniden train/validation kaynak kökleriyle exact q,
   pose, group ve konum 2 mm + yönelim 1° yakın poz taramasıyla doğrulanır.
   C1-02 test ayrımı hashli kabul/leakage manifestine dayanır; yeniden açılmaz.
3. C1-01 her solver için 12.000 query × 2 deadline × 5 pass tekildir. Ham
   query_id, query_group_id, q_current, hedef konum/quaternion, query-list SHA,
   robot/TCP hashleri, joint sırası ve tolerans profilleri aynı sorguya bağlanır.
   Float64 q_current/hedef listeleri aynen eşleşir; farkı yuvarlayarak gizleme yoktur.
4. Farklı sorgu, eksik/tekrarlı satır veya soy çakışması teknik FAIL'dir;
   eşleşmeyen satırlar çıkarılarak paired CI üretilmez. Özetler raw yerine geçmez.

Tarihsel üretici alt küme köklerinin ayrı olduğunu bildirir. Aynı kökün birden
fazla tanı etiketi varsa bir query_id tekil paydada yalnız bir kere bulunur;
alt küme üyelikleri çoklu olabilir. Aynı kök/trajectory'den gelen bütün satırlar
aynı bootstrap kümesindedir. Gerçek beklenmedik kaynak çakışması birleştirilerek
gizlenmez; audit başarısız olur ve gerekçeli protokol revizyonu gerektirir.

## Birincil H2 ve istatistik

E-C05/FK_TANH − E-C03/Q, aynı seed ve query_id, aynı Profile A denetleyicisi.
Hedef zor dağılımların eşit ağırlıklı ortalamasında ≥+2 yüzde puanı,
main kümede ≥−1 yüzde puanı farktır. FK_TANH kayıp ve sınırlı başlığı birlikte
değiştirir; sonuç bunlardan yalnız birine atfedilmez. C1-04 conditioned Q
validation eşliği bilinen tekrar kontrolüdür, bağımsız örneklem değildir.

Her seed/query için tek 0/1 geometri sonucu. Beş süre geçişinde neural q ve
geometri sonuçları aynen tekrarlanmalı; farklıysa teknik inceleme gerekir,
başarılı geçiş seçilmez. Bootstrap N, süre geçişleri veya eğitim seedleri ile
çarpılmaz. Eksik gözlem yok sayılmaz veya başarı sayılmaz: tam envanter şarttır.
Sonlu olmayan/limit dışı/geometri başarısızlığı gözlenen sıfır başarıdır;
teknik olarak işlenmemiş satır farklıdır ve T-C05'i engeller.

PCG64 seed=2026100806, 10.000 bootstrap tekrarı, iki taraflı %95 percentile
[2,5;97,5], NumPy linear quantile. Root grupları alt küme üyelik birleşimine
göre tabakalanır; her tabakada grup sayısı kadar yerine koyarak çekilir.
Grup içindeki bütün sorgular, iki kol ve üç seed aynı örnekleme ağırlığını
paylaşır. Alt küme farkı satır ağırlıklı ortalamadır; zor fark boundary ve
singularity farklarının 0,5/0,5 ortalamasıdır. Main, boundary, singularity,
zor ortalama için seed bazında ve üç sabit seed eşit ortalamasıyla nokta/CI verilir.
Seed min/max ve ayrı değerler raporlanır; üç seed için popülasyon CI iddiası yoktur.

Karar, üç sabit seed ortalamasının CI'larından, teknik kapılar geçtikten sonra:

| Karar | Önceden sabit koşul |
|---|---|
| Desteklendi | Zor fark CI alt sınırı ≥0,02 VE main fark CI alt sınırı ≥−0,01 |
| Reddedildi | Zor fark CI üst sınırı <0,02 VEYA main fark CI üst sınırı <−0,01 |
| Belirsiz | Yukarıdaki iki koşul da sağlanmıyor |
| Teknik olarak belirsiz | Eksik veri, kimlik/soy/denetleyici hatası; T-C05 FAIL/NOT_RUN |

Nokta hedefe ulaşsa bile CI koşulları geçmeden destek denmez. Tam sıfır başarı
empirik bootstrap'ta [0,0] üretebilir: bu gözlenen sabit örneklemde hedeflenen
artışın sağlanmadığını gösterir, gerçek popülasyon oranının tam sıfır olduğunu
kanıtlamaz. Bootstrap yalnız bu veri ve sabit seedler için belirsizlik sunar.
Olumsuz H2, eksiksiz teknik yürütmede araştırma kapanışını engellemez.

İkincil karşılaştırmalar COMPARISON_MATRIX.md'de sabittir; aynı bootstrap
yöntemiyle keşifsel nominal %95 CI. Çoklu kıyas düzeltmesi yapılmadığı açıkça
yazılır; bunlardan birincil H2 veya üstünlük seçilmez. E-C04/E-C05 FK kontrolü
kendi eşli kısa eğitiminden alınır, E-C03'ün uzun FK kontrolüyle değiştirilmez.

## Doğrulayıcı, payda ve hata analizi

Model girdisi yalnız normalize hedef position_m, wxyz quaternion ve limitlerle
normalize q_current (13 float32); normalizasyon yalnız C1-02 train'den.
q_target/teacher sonucu girdi değildir. q çıktısı değiştirilmez/clamp edilmez.
Her satırda shape/finite/limits ayrı; bağımsız NumPy FK (Pinocchio veya Torch
çıktısı kopyalanmadan). Profile A: ≤0,002 m ve ≤1°; B: ≤0,001 m ve ≤0,5°.
Deadline başarısı geometri başarısından ayrıdır; collision=NOT_CHECKED.

Zorunlu raw: run/model/checkpoint SHA, seed, query/group/source/subset/mode,
query/robot hashleri, pass, q_raw, shape/finite/limits, poz m/geodezik derece,
Profile A/B, timeout10/50, hata sınıfı, elapsed_ns, label/teacher durumu.
NaN/Inf JSON'da null + açık failure alanıyla saklanır; sıfıra çevrilmez.
Hata önceliği INVALID_SHAPE → NONFINITE → JOINT_LIMIT → POSE_TOLERANCE;
timeout ayrı bayrak, teknik exception ayrı terminal audit hatasıdır.

Her yöntem/seed için bütün sorgu paydasında A/B, nonfinite, limit, timeout,
çözülememe oranları ve N; main/local/wide/boundary/singularity kırılımları.
Poz/yönelim median/P95/P99 yalnız hesaplanabilir (finite + limit içi) satırlarda,
coverage ve eksik adetleriyle birlikte; başarılı-satır istatistiği ayrıca ve
tam-payda başarı oranıyla yan yana. Geçersiz q için FK metriği null kalır.
Benchmarkta q_target kaynak tanığıdır, teacher etiketi değildir; label ve
teacher başarı alanları NOT_APPLICABLE'dır. C1-02 test teacher oranları final
benchmark sonucu gibi kullanılmaz; geçmiş eğitim etiket eksikliği açıklanır.

Ön tanımlı failure analizi: her aile/mode için oran; en kötü 20 query/root
(önce poz, sonra yönelim; ters sıra ayrı tablo), en büyük 20 yönelim hatası,
normalize limit uzaklığı <0,02; sigma_min ≤0,00727741160967353 (ℓ=0,9015 m);
train position z-score |z|>3 dağılım kayması tanısı. Bunlar keşifseldir, yeni
filtre veya başarı eşiği değildir. Sonuçtan sonra model/λ değiştirilmez.

## Süre ve baseline karşılaştırmasının sınırı

Windows CPU tek sorgu, tek Torch/BLAS thread, inference mode, 21 model sırayla.
Model yükleme ve 20 warmup çağrısı ayrı ölçülür; warmup yalnız 10 mevcut
validation tanığı iki kez. Her model 12.000 sorguyu beş aynı-sıralı geçişte
işler. Uçtan uca süre input hazırlığı + model ileri geçiş + raw q dönüşümü +
bağımsız doğrulama; disk okuma/yazma ve model yükleme hariç. Tek geçişteki
elapsed_ns hem 10 hem 50 ms eşiğiyle etiketlenir; sahte preemptive timeout yoktur.
Bütün ve başarılı sorgular için P50/P95/P99 ayrı; başarı yoksa süre null/N=0.
GPU batch throughput NOT_RUN; CPU süreleriyle karıştırılmaz.

C1-01 Ubuntu/ROS, farklı IPC/kaynak/warmup koşulları nedeniyle yalnız tarihsel
betimsel süre tablosudur. Aynı hostta beş baseline yeniden çalıştırılmadıkça
hız üstünlüğü/H1 CI verilmez. Geometri karşılaştırmasında historical timeout
nedeniyle adayı olmayan kayıt başarısız olarak korunur; süre kısıtı görünürdür.
Her 10/50 ms profili ayrı, beş geçiş query başına ortalanır; 120.000 ölçüm
12.000 bağımsız sorgu yerine kullanılmaz. Baseline-neural farkları keşifsel,
eşli query-root CI ile ayrı raporlanır; birincil 0/1 H2 koduna beş tekrar
çoğaltılarak veya kesirli gözlem zorlanarak sokulmaz.

## Kesinti, kanıt ve kapanış

Her komut argv/exit/stdout/stderr, başlangıç/bitiş ve hashle saklanır.
Raw LOCAL_ONLY: data/generated/C1-06/<run>/; küçük özet/CI/failure/audit/log
experiments/C1-06/stage2/<run>/. Satır sayısı, byte, SHA, erişim ve uzak arşiv
durumu yazılır. Ham veri silinmez; interrupted run ayrı dizinde korunur.
Tek final kampanyası, önceden dondurulmuş beş ölçüm geçişini içerir.
Kesintide model/seed/son query/pass kaydedilir; yeni attempt/ADR ile teknik
düzeltme açıklanır, eski ve yeni kanıt sessiz birleştirilmez veya overwrite edilmez.

T-C05 PASS: korunmuş ön kayıt, 21 checkpoint × tam sorgu envanteri, beş baseline
raw join, bağımsız FK, paired bootstrap, gerekçeli H2, failure raporu ve hashli
taşınabilir C1-07 devri. C1-07/G1/v1.0.0 bu görevin yetki/kapanış alanı değildir.
Kinematik başarı fiziksel güvenlik/çarpışmasızlık veya erişilemezlik kanıtı değildir.
