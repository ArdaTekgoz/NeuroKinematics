# ROADMAP EXTENSION — Fiziksel yürütme ve ürünleştirme

Roadmap r3 · 11 Ekim 2026 · Durum PLANLANDI / NOT_STARTED · Tahmin: Platform ve test kapsamı netleşince tahminlenecek.
Yazılım hedef sürümü belge revizyonundan ayrıdır; henüz release yapılmadı.

Kaynak PDFnin sekiz PE başlığı korunur; PE-05 ve PE-07 ön koşul/kapanış
olarak bölünür. PE-04 fiziksel hareketi, kalibrasyon/HIL/interlocklardan sonra gelir.
Bu sıra düzeltmesi ADR-027de gerekçelidir. Giriş: G-MP; arayüz uzantısında ayrıca G3.
TK56 fiziksel referans adayı, KUKA algoritmik referans olarak kalır.
Donanım mevcudiyeti ve ölçüm/stop değerleri henüz doğrulanmadı; toplam saat ve
takvim bu yüzden üretilmedi. Her task öncesi donanıma özel implementation specification gerekir.

PE-02 mock iletişim kabulü sonraki işi açar; gerçek I/O kabulü PE-03/06da kapanır.
PE-03/05A/06 kısıtlı bench/commissioning çalışmalarını kapsar; tam trajectory
yürütme PE-07Bden önce açılmaz. Sanal başarı fiziksel kabul yerine kullanılmaz.

## Uygulama sırası

| Sıra | Görev | Ön koşul | Saat |
|---|---|---|---|
| 1 | [PE-07A — Fiziksel kapsam ve güvenlik gereksinimleri](../tasks/PE-07A.md) | MP-05 | TBD |
| 2 | [PE-01 — Digital twin yürütme ön doğrulaması](../tasks/PE-01.md) | PE-07A | TBD |
| 3 | [PE-02 — ROS 2 ve donanım soyutlama](../tasks/PE-02.md) | PE-01 | TBD |
| 4 | [PE-03 — Embedded controller ve commissioning arayüzü](../tasks/PE-03.md) | PE-02 | TBD |
| 5 | [PE-05A — Temel kalibrasyon ve robot kimliği](../tasks/PE-05A.md) | PE-03 | TBD |
| 6 | [PE-06 — Hardware-in-the-loop ve hata yönetimi](../tasks/PE-06.md) | PE-05A | TBD |
| 7 | [PE-07B — Yürütme kilitleri ve fiziksel kabul](../tasks/PE-07B.md) | PE-06 | TBD |
| 8 | [PE-04 — Kapalı çevrim fiziksel trajectory yürütme](../tasks/PE-04.md) | PE-07B | TBD |
| 9 | [PE-05B — Sim-to-real doğruluk ve tekrarlanabilirlik](../tasks/PE-05B.md) | PE-04 | TBD |
| 10 | [STUDIO-EXT — Fiziksel robot operasyon ekranları](../tasks/STUDIO-EXT.md) | S3-05, PE-05B | TBD |
| 11 | [PE-08 — Dağıtım ve güvenilirlik ürünleştirme](../tasks/PE-08.md) | STUDIO-EXT | TBD |

## Görev bazında kabul

### PE-07A — Fiziksel kapsam ve güvenlik gereksinimleri

Gerçek hareketten önce referans platformu, risk sınırlarını ve donanım sorumluluğunu belirlemek.

**Gereksinim:** REQ-PE07A. **Test:** T-PE07A.

**Çıktı:** TK56 adayının gerçek actuator/encoder/driver/controller, joint/velocity/acceleration limitleri, güç/IO, acil durdurma ve mekanik sınır envanteri; mevcut olmayan özelliği mevcut sayma. Uygulama, ROS, embedded ve driver sorumluluk matrisi; E-stop/disable/limits/watchdog/fault/startup/shutdown/reset gereksinimleri. İlk enerjilendirme ve sınırlı commissioning test planı; fiziksel test alanı, operatör yetkisi, durdurma ölçütleri ve bağımsız stop yolu. Controller rate, feedback age, watchdog timeout, tracking threshold, stop-time/distance ve calibration hedeflerini donanım kaynaklarıyla ön kaydet. Sayısal değerler henüz yoksa BLOCKED_SPEC. PE-02/03 teknik platform ve transport seçimi için implementation specification; STM32/CAN öneridir, zorunlu evrensel NK bağımlılığı değildir.

**Kabul:** T-PE07A: gereksinim→sorumlu katman→test eşlemesi; kritik limit/zaman/stop değerleri kaynaktan doğrulanmış, açık eksikler listelenmiş. Bu task fiziksel uygunluk/safety sertifikası üretmez; PE-07B yürütme interlock kabulü olmadan PE-04 çalışmaz.

**Başarısızlıkta:** Donanım veya ölçüm imkânı yoksa ilgili fiziksel task WAITING_HARDWARE; simülasyon sonucu yerine konmaz.

**Kanıt:** `experiments/PE-07A/`; PLANLANDI / NOT_RUN.

### PE-01 — Digital twin yürütme ön doğrulaması

Plan ve trajectory için görüntüden bağımsız makine okunur yürütme ön doğrulaması kurmak.

**Gereksinim:** DT-REQ-01, DT-REQ-02, DT-REQ-03, DT-REQ-04, DT-REQ-05. **Test:** T-PE01.

**Çıktı:** Aynı RobotModel/scene/trajectory revisionu; simüle state kaynağı, ileride timestampli q_actual adapterı. q/dq/ddq, limit, self/environment collision, süreklilik, son pose, solver kabulü ve singularity kontrolleri. VALID/VALID_WITH_WARNING/INVALID ve nedenleri; zorunlu kontrolün UNKNOWN sonucu authorization vermez. Plan hash + scene hash + başlangıç state/zamanını validation raporuna bağla; değişirse rapor stale. Robot olmadan tüm sanal akış ve invalid testleri; görüntü artifactları sayısal rapordan türetilsin.

**Kabul:** T-PE01: geçerli/limit aşımı/çarpışma/kopukluk/stale-scene/yanlış robot testleri; zorunlu kusurlu trajectory execution adayı olamaz. VALID yalnız yazılım ön koşulu; ayrıca calibration, hardware readiness, HIL ve fiziksel yetkilendirme gerekir.

**Başarısızlıkta:** Digital twin fiziksel ölçümün veya donanım stop zincirinin yerini alamaz; sim-to-real iddiası PE-05Bde ölçülür.

**Kanıt:** `experiments/PE-01/`; PLANLANDI / NOT_RUN.

### PE-02 — ROS 2 ve donanım soyutlama

NK ile controller arasında standardize command/state ve adapter sınırı kurmak.

**Gereksinim:** HW-REQ-01, HW-REQ-02, HW-REQ-03, HW-REQ-04, HW-REQ-05. **Test:** T-PE02.

**Çıktı:** NK→ROS 2→ros2_control→hardware adapter→transport akışı; sürümler/OS/driver uyumluluğunu uygulama başında resmi belgeler ve kurulumla doğrula. Joint position/velocity/controller/fault state ve trajectory/position/velocity/control command; sıra, birim, timestamp, sequence ID. DISCONNECTED/CONNECTING/CONNECTED/READY/EXECUTING/FAULT; reconnect, stale/out-of-order feedback, cancel ve diagnostics. Önce fake hardware read/write; core içine CAN/STM32 kodu sokma. Gerçek I/O doğrulaması PE-03/06 kanıtıyla kapanır.

**Kabul:** T-PE02: mock bidirectional roundtrip, sıra/birim/frame, timestamp ve stale message reddi; state machine testleri PASS. Gerçek state/command şartı fake testle sağlandı sayılmaz; fiziksel kabul PE-06 bağımlı alt kontrol olarak açık tutulur.

**Başarısızlıkta:** Bağlantı kurulması motion authorization değildir; otomatik reconnect hareketi başlatmaz.

**Kanıt:** `experiments/PE-02/`; PLANLANDI / NOT_RUN.

### PE-03 — Embedded controller ve commissioning arayüzü

Encoder, homing, motor setpoint ve yerel fault/watchdog işlevlerini controller katmanında doğrulamak.

**Gereksinim:** EMB-REQ-01, EMB-REQ-02, EMB-REQ-03, EMB-REQ-04, EMB-REQ-05, EMB-REQ-06. **Test:** T-PE03.

**Çıktı:** Platform kararına göre reference embedded firmware veya mevcut controller adapterı; firmware/config sürüm/hash. Encoder resolution/direction, gear ratio, zero offset, joint direction, motor ID ve home mekanizması. BOOT/INITIALIZATION/HOMING/READY/EXECUTING/STOPPED/FAULT; actuator enable/disable; watchdog PCden bağımsız. İlk test fake loads/bench; enerjili commissioning yalnız PE-07A planı ve temel stop/disable kanıtı ile. Serbest/full trajectory execution burada yapılmaz.

**Kabul:** T-PE03: state acquisition ve config dönüşümleri, setpoint işleme, homing tekrarı, yerel watchdog/fault propagation; donanım ön kayıtlı zaman sınırlarında. State/command/diagnostics çift yönlü fiziksel bench kanıtı; NK yüksek seviye süreçten bağımsız controller davranışı.

**Başarısızlıkta:** Bir MCU markası ürün core gereksinimi yapılmaz; eksik real-time/stop ölçümü PASS değil.

**Kanıt:** `experiments/PE-03/`; PLANLANDI / NOT_RUN.

### PE-05A — Temel kalibrasyon ve robot kimliği

Kapalı çevrim trajectory yürütmeden önce gerçek joint ve model eşleşmesini doğrulamak.

**Gereksinim:** REQ-PE05A. **Test:** T-PE05A.

**Çıktı:** Encoder zero/scale/direction, gear ratio/homing offset, joint order/sign ve physical model identity. Base/TCP/tool, nominal link/axis ölçüleri ve başlangıç calibration; bağımsız ölçüm yöntemi/belirsizlik/cihaz bilgisi. Robot_revision/URDF_hash/firmware/calibration_version ile kalibrasyon paketi; değişiklikte eski validation ve model stale. Sınırlı commissioning hareketleri PE-07A planına bağlı; tam çalışma zarfı sim-to-real değerlendirmesi PE-05B.

**Kabul:** T-PE05A: işaret/sıra/ölçek/offset ve başlangıç pose eşliği ön kayıtlı mühendislik toleransında. Sadece aynı URDF ile iki FK sonucu karşılaştırması fiziksel kalibrasyon sayılmaz; bağımsız ölçüm gerekir.

**Başarısızlıkta:** Ölçüm veya tolerans kaynağı eksikse full execution yok; nominal URDF fiziksel doğruluk kanıtı değildir.

**Kanıt:** `experiments/PE-05A/`; PLANLANDI / NOT_RUN.

### PE-06 — Hardware-in-the-loop ve hata yönetimi

Tam fiziksel yürütmeden önce haberleşme ve controller hatalarının davranışını ölçmek.

**Gereksinim:** REQ-PE06. **Test:** T-PE06.

**Çıktı:** Disconnect, ROS loss, delayed feedback, corrupted/replayed message, missing/inconsistent encoder, joint limit request, abort, controller restart, PC restart, NK crash ve driver fault enjeksiyonu. Her fault için detection/propagation/response/reset; INFO/WARNING/RECOVERABLE/CRITICAL sınıfları. Süre damgalı watchdog/stop ölçümleri; simulation vs actual HIL kaynakları ayrı. Bench/restricted commissioning dışında hareket açma. PE-02 gerçek read/write ve PE-03 fault davranışlarını end-to-end doğrula.

**Kabul:** T-PE06: tüm zorunlu fault senaryoları ön kayıtlı response/latency/stop koşullarına uygun; sessiz devam/otomatik yeniden hareket yok. Fiziksel HIL çalıştırılmadıysa HIL PASS yok; sadece mock katmanı PASS olarak tutulur.

**Başarısızlıkta:** Kritik fault koşulu karşılanmazsa PE-07B/PE-04 kapalı; uygulama UI stopuna güvenerek kusuru örtme.

**Kanıt:** `experiments/PE-06/`; PLANLANDI / NOT_RUN.

### PE-07B — Yürütme kilitleri ve fiziksel kabul

Tam trajectory yürütme öncesi donanım ve yazılım ön koşullarını kilitlemek.

**Gereksinim:** REQ-PE07B. **Test:** T-PE07B.

**Çıktı:** Physical E-stop, actuator disable, hardware/software limits, driver faults, watchdog/comm timeout, controlled reset/startup/shutdown kanıtları. Readiness authorization: model/calibration/scene/trajectory/firmware identity, fresh state, twin valid, HIL accepted ve operatör yetkisi. UI acil durdurması ayrı kolaylık; fiziksel E-stop yerine geçmez. PC kapanması ve bağlantı kaybında yerel durdurmayı ölç. Kilit atlama/unknown-required-check/stale-authorization negatif testleri.

**Kabul:** T-PE07B: ön kayıtlı durdurma ve interlock testlerinin tümü PASS; eksik zorunlu bilgiyle execution 0. Bu mühendislik kabulü sertifikasyon iddiası değildir; test kapsamı ve sorumlular açık.

**Başarısızlıkta:** Critical interlock başarısızsa full movement disabled; gerçek donanım değerlendirmesi olmayan sonucu uygunluk belgesine dönüştürme.

**Kanıt:** `experiments/PE-07B/`; PLANLANDI / NOT_RUN.

### PE-04 — Kapalı çevrim fiziksel trajectory yürütme

Planlanan ve ölçülen gerçek hareketi karşılaştırıp yürütmeyi izlemek.

**Gereksinim:** REQ-PE04. **Test:** T-PE04.

**Çıktı:** Yalnız geçerli twin raporu ve PE-07B authorization ile; q_commanded/q_actual timestamp eşlemesi ve tazelik kontrolü. Tracking error, progress, completed/aborted, aşırı hata, controlled stop, fault propagation ve loglar. Ön kayıtlı sınırlı hareket profiliyle başlayıp kabul sonrası zarfı genişlet; farklı zarf ayrı profil. Mean/max tracking, final joint/Cartesian error, duration ve faults; commanded bilgi actual diye kullanılmaz.

**Kabul:** T-PE04: completion/abort/stop/target tolerance ön kayıtlı donanım ve calibration profiliyle PASS. Gerçek joint ölçümü olmadan closed-loop execution iddiası yok; final Cartesian doğruluk için bağımsız ölçüm ayrı belirtilir.

**Başarısızlıkta:** Stale state veya aşırı tracking error kontrollü stop/fault; hedefe varmış animasyon fiziksel başarı sayılmaz.

**Kanıt:** `experiments/PE-04/`; PLANLANDI / NOT_RUN.

### PE-05B — Sim-to-real doğruluk ve tekrarlanabilirlik

Nominal/kalibre modelin fiziksel trajectory ile farkını ölçmek.

**Gereksinim:** REQ-PE05B. **Test:** T-PE05B.

**Çıktı:** Aynı hareketin simülasyon/fiziksel joint/pose kayıtları; bağımsız pose ölçümü ve ölçüm belirsizliği. Joint/position/orientation error ve repeatability; farklı workspace/yük/tool profilleri yalnız tanımlı kapsamda. Calibration güncellemesi gerekirse yeni revision; dataset/model/twin/trajectory bağımlılıklarını yeniden doğrula. Final veri ile kalibrasyon ayarlama yok; calibration-fit ve held-out ölçüm setlerini ayır.

**Kabul:** T-PE05B: ön kayıtlı fiziksel tolerans ve tekrar protokolü; sim-real farklar kapsam/belirsizlikle raporlu. Yeni kalibrasyonun regression ve stale-artifact politikası PASS; eski frozen KUKA kaynakları korunur.

**Başarısızlıkta:** Tolerans dışıysa physical support profili açık kalır; sadece simülasyon hatasını raporlayarak fiziksel fark saklanmaz.

**Kanıt:** `experiments/PE-05B/`; PLANLANDI / NOT_RUN.

### STUDIO-EXT — Fiziksel robot operasyon ekranları

Mevcut Studioyu gerçek robot readiness ve execution izlemeyle genişletmek.

**Gereksinim:** REQ-SX04. **Test:** T-SX04.

**Çıktı:** Connect/robot readiness/calibration/controller/hardware compatibility ve canlı q/velocity/TCP/fault/target/actual/tracking error. TARGET→SOLVE→PLAN→VALIDATE→SIMULATE→AUTHORIZE→EXECUTE→MONITOR; ayrı stop/abort ve fault görünümü. Sadece backend servisleri; GUI içinde CAN/STM32/servo iletişimi yok. Yanlış robot, stale state, eksik calibration, expired validation, disconnect ve race durumlarında hareket düğmesi ve backend reddi birlikte test.

**Kabul:** T-SX04: frontend ve backend aynı authorization kararı; zorunlu koşul yokken execute mümkün değil; actual/commanded görsel ayrımı doğru. Ayrı yeni Studio core/fazı oluşturulmaz; S3 regresyonları korunur.

**Başarısızlıkta:** Arayüzde görünür yeşil durum fiziksel safety kanıtı değildir; backend kilidi UI üzerinden aşılamaz.

**Kanıt:** `experiments/STUDIO-EXT/`; PLANLANDI / NOT_RUN.

### PE-08 — Dağıtım ve güvenilirlik ürünleştirme

Referans fiziksel entegrasyonu sürümlenmiş tekrar kullanılabilir platforma dönüştürmek.

**Gereksinim:** REQ-PE08. **Test:** T-PE08.

**Çıktı:** Robot model plugin/hardware adapter/calibration/controller config sınırları; driver/solver/planner/simulator/calibration extension sözleşmeleri. Automated regression, endurance, crash recovery, version compatibility, diagnostics, deployment ve dependency/SBOM; telemetry kapsamı açıkça seçilir. Desteklenen robot/platform/yük/hız/scene/firmware profili; install/update/rollback/backup/restore doğrulaması. Endurance süresi, fault oranı ve deployment kabul hedefleri kaynak planına göre test öncesi dondurulur; TK56 başarısını tüm robotlara genelleme.

**Kabul:** T-PE08: ön kayıtlı dayanıklılık ve recovery testleri PASS, bağımsız clean deployment ve compatibility matrisi. Fiziksel ürün kapsamı G-PE kararıyla sınırlı; tüm PE kanıtları ve açık riskler bağlı; sertifikasyon iddiası yok.

**Başarısızlıkta:** Donanım bütçesi/ekip/bağımlılık belirsizken süre veya ürün hazır sözü verilmez; kritik reliability kusuru gate engeli.

**Kanıt:** `experiments/PE-08/`; PLANLANDI / NOT_RUN.

## Faz kapısı

**G-PE:** gerçek state/command, calibration, HIL, yerel fault/stop,
closed-loop tracking, bağımsız sim-to-real ölçümü, Studio operasyon ve endurance/
clean deployment ön kayıtlı şartları kapalı. Her ara PE gate ayrı kabul ister.
Donanım yoksa WAITING_HARDWARE; test yoksa NOT_RUN. Digital twin VALID,
fiziksel güvenlik sertifikası veya tek başına execution izni değildir.

Tüm işler için gereksinim → değişiklik → test → ham kanıt bağı tutulur.
Uygulamada protokol/seed/kimlikler sonuç görülmeden dondurulur; başarısız kayıtlar
korunur. Yukarıdaki süreler tahmindir; GPU bekleme, donanım temini ve kullanıcı
testi takvimini kapsamaz. İlk iki görevden sonra yalnız tahminler güncellenir.

[Ana roadmap](MASTER_ROADMAP.md) · [Kaynak eşlemesi](SOURCE_MAPPING_r3.md)
· [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).
