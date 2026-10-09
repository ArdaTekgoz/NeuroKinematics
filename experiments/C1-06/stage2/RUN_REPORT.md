# C1-06 bağımsız nihai değerlendirme çalışma kaydı

Kimlik: RUN-20261008-C106-STAGE2
Durum: COMPLETE / T-C05 PASS / H2 REJECTED
Görev ve gereksinim: C1-06 / REQ-C04, REQ-C05
Tarih ve sorumlu: 8–9 Ekim 2026, Codex; proje sahibi Arda Tekgöz.
Yazılım hedefi v1.0.0, belge r3; G1 ve sürüm etiketi verilmedi.

## Soru ve değişiklik

Ön kayıtlı üç seedli FK_TANH ailesi, aynı seedli supervised Q kontrolüne göre
zor alt kümelerde ≥+2 yüzde puanı katkı sağlıyor mu; main düşüşü ≤1 yüzde
puanında mı? Kullanıcı Stage1 commit/push sonrasında açıkça onayladı.
Model/λ/eşik/seed değiştirilmeden değerlendirme wrapper'ı tamamlandı ve
final testten önce preflight.json ile kaynak/test hashleri sabitlendi.
Stage1 kaynak protokol ve istatistik/karar kodu değişmedi.

| Gereksinim | Uygulama | Test | Ham kanıt / karar |
|---|---|---|---|
| REQ-C04 | 21 sabit checkpoint, üç seed, ayrı eşli ablasyonlar | 210 sabit tanık + 59 sentetik test | preflight.json, runtime-tests.xml |
| REQ-C05 | Sorgu/soy/FK ve baseline raw join | 12.000 query + 600.000 raw; sıfır sızıntı | final-001/identity.json |
| REQ-C04/05 | Tam payda, beş geçiş, bağımsız FK | 1.260.000 neural raw; 252.000 tekil model/query | final-001/evaluation.json, raw-manifest.json |
| REQ-C04/05 | Root paired bootstrap, failure ve timing | 10.000 bootstrap tekrarı; aynı query/seed | final-001/results.json; H2 REJECTED |
| REQ-C05 | Bütün kayıtların sayı/SHA ve karar kontrolü | Ham satır denetimi, tekrar eşliği, iki FK | final-001/acceptance.json: T-C05 PASS |

## Tekrar üretim

Stage1 commit `075478545a30373db2a4ac64434f2db678df050e`; Stage2 yürütücü
`e97425d`. Final açılışı opening.json: 8 Ekim 2026 19:17:34 UTC
(22:17:34 Europe/Istanbul); kimlik kapısı 19:20:29 UTC’de tamamlandı.
Ölçüm/analiz 8 Ekim, belge ve Git teslimi 9 Ekim 2026’dır. Ön kayıt ve checkpoint hashleri ../input-hashes.json,
../SHA256SUMS, ../config.json; kod/test freeze preflight.json içindedir.
Son teslim commit'i kendine referans vermeden Git geçmişinden alınır.

Windows 11 Pro, AMD Ryzen 7 250 (8 core/16 logical), Python/Torch/NumPy
mevcut locked Pixi + c103 overlay; tam sürümler preflight.json. Tek Torch/BLAS
thread, tek sorgu CPU. hardware.json ve resource-observations.jsonl gerçek
gözlemleri saklar; gözlenen OS process peak RAM sürekli profil veya sabit RAM
tavanı değildir. Yeni temiz ortam ve yeni eğitim NOT_RUN. GPU/Linux/fiziksel
robot NOT_RUN, etkin insan emeği NOT_MEASURED.

Gerçek komut sırası [COMMANDS](COMMANDS.md), argv/exit/stdout/stderr ../commands.
Neural kampanya duvar süresi 510.277 s. Model yükleme,
warmup ve model başına ölçüm duvar süreleri evaluation.json içinde ayrı.
Başlangıçtaki STATUS/TRACEABILITY audit değişiklikleri, kullanıcı PDF/Word
ve AUDIT-20261003 klasörü commit dışında korunmuştur.

## Test ve ham kanıt

322 Stage1 girdi ve 36 ön kayıt dosyası hash kapısı PASS. C1-05 eski witness
180 q/FK değerini birebir tekrarladı. Eski C1-04 wrapper çağrısı .gitattributes
tarihsel SHA nedeniyle exit1; log korunur. Yeni doğrudan preflight 21 modelin
210 tanığını, asıl robot/config/normalizasyon/weight hashleriyle PASS doğruladı.
59 sentetik/negatif test PASS (fail0/error0/skip0). Testler final sorgu kullanmadı.

Nihai query/root 12.000/12.000. Main 5.000 local + 5.000 wide; boundary
500+500, singularity 500+500. 20.400 train/validation çift kaydının kaynak
kökleri incelendi; sekiz yakın konum adayı yönelim kontrolüne girdi, yakın
pose sızıntısı sıfır. C1-02 test shardları önceki kabul/split hashleriyle
korundu ve açılmadı; 3.600 çift ana benchmark paydasına katılmadı.

Beş baseline 600.000 exact query/target/q_current bağını geçti. Aynı bağımsız
FK'de A/B ve joint-limit kararları geçmişle tamamen aynı. En büyük tarihsel
platform residual farkları identity.json'da; eşik veya raw değiştirilmedi.
Neural 21 × 12.000 × 5 = 1.260.000 satır; bütün beş geçişte q/geometri birebir.
İlk geçişteki 252.000 model/query sonucu tekrar NumPy FK ile denetlenip geçerli
q'larda ayrıca Pinocchio ile G0 toleranslarında karşılaştırıldı. Fail satırları
paydada kaldı, invalid q clamp edilmedi. Eksik/çift gözlem yok.

Yeni raw dosyalar data/generated/C1-06/final-001 altında, toplam
2,331,485,455 bayt. Her dosyanın rows/bytes/SHA/access durumu
final-001/raw-manifest.json'da. C1-01 özgün yaklaşık 1,112 GB raw ve bütün
weights/shardlar LOCAL_ONLY; uzak arşiv NOT_CONFIRMED. Küçük rapor/log/kararlar
Git'te; büyük dosyalar Git'e eklenmedi.

## Sonuç ve yorum

**T-C05 PASS**, çünkü bütün ön kayıtlı modeller/sorgular/baselinelar, bağımsız
denetim, paired istatistik, başarısızlık analizi ve raw kanıt tamamlandı.
**H2 REJECTED**. Ayrıntılı [sonuç tablosu](RESULTS.md): üç seed
ortalamasında main +0.000 [+0.000, +0.000] yp;
zor eşit ağırlık +0.000 [+0.000, +0.000] yp.
Nokta ve CI ön kayıtlı karar ağacına aynen uygulandı. Bu araştırma kapanışı,
başarılı operasyonel IK veya fiziksel robot güvenliği kabulü değildir.

İkincil FK−Q, FK_LIMIT−eşli FK, FK_TANH−eşli FK ve C1-04/Q kontrolü
ayrı keşifsel CI ile saklandı. Farklı gerçekleşen epoch/compute, sınırlı üç
seed, FK_TANH'da kayıp+head bileşik değişimi, eğitim/benchmark dağılım farkı
ve teacher seçimi genelleme sınırlarıdır. Empirik [0,0] bootstrap popülasyonda
mutlak sıfır başarıyı kanıtlamaz. Kötü sonuçlar eşik değişikliğiyle gizlenmedi.

Windows CPU uçtan uca süreleri ve Ubuntu/ROS tarihsel süreleri ayrı kapsamda.
Aynı platform eş ölçüm yapılmadığı için hız üstünlüğü iddiası yok.
Başarılı-altküme boşsa süre null/N=0. Collision NOT_CHECKED; solver failure
erişilemezlik kanıtı değildir. Bilinen no-candidate baseline kayıtlarının
adapter failure_class yorumu RESULTS.md'de açıklandı; timeout ve common_status
ayrı korunur, candidate yokluğu malformed solver output diye sunulmaz.

## Sonraki adım

[C1-07 devri](final-001/C1-07-handoff.json): seçilmiş FK_TANH ailesinin üç
checkpoint kimliği, query/config/normalizasyon, H2/CI, raw manifest/erişim ve
yeniden üretim tarifi hazır. C1-07 model kartı ve G1 kabulü ayrı görevdir,
başlatılmadı. Yeni model/λ denemesi için yeni config ve gerektiğinde yeni
bağımsız final test sürümü gerekir; bu final sonuç tuning amacıyla yeniden
kullanılmaz. Orijinal Stage1 ve bütün başarısız/başarılı komut logları korunur.


## 9 Ekim 2026 · Teslim denetimi

Ölçümler ve teknik kabul 8 Ekim’de tamamlandı; 9 Ekim devamında yeni model
veya final ölçümü çalıştırılmadı. Kaydedilmiş kabul/ham kanıt korunarak
STATUS/TRACEABILITY/roadmap/görev kapanışı ve C1-07 devir paketi tamamlandı.
Son byte/SHA, komut logu, rapor bağlantısı ve Q tekrar kontrolü
`python scripts/check_c106_bundle.py --check` ile ayrı, salt okunur denetlenir.
Bu komutun gerçek exit/stdout/stderr kaydı ../commands altında tutulur.

İlk teslim checker'ı JSON tanı dosyasında JSONL rows alanı beklediği için
exit1 verdi; freeze dosyası üretilmedi. Yalnız teslim checker şema okuması
düzeltildi, ilk log korundu ve ikinci freeze denemesi ayrı adla kaydedildi.
Bu belge denetimi düzeltmesi model/istatistik protokolünü veya ham ölçümü
değiştirmez; yeni final çalıştırması yapılmadı.
