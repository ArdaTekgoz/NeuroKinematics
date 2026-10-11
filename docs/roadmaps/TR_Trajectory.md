# ROADMAP EXTENSION — Studio trajectory üretimi

Roadmap r3 · 11 Ekim 2026 · Durum PLANLANDI / NOT_STARTED · Tahmin: 52–82 etkin saat.
Yazılım hedef sürümü belge revizyonundan ayrıdır; henüz release yapılmadı.

S3-02 generic GUI/FK kabuğu sonrası, Motion Planning öncesi. Joint-space
point-to-point/waypoint trajectory üretir. IK çözmez, engelden kaçan yol aramaz,
motor kontrolü yapmaz. H2-04 solver rollout testlerini tamamlar; onların yerine geçmez.
Kaynak PDFdeki T-S01–10, mevcut Studio testleriyle çakışmasın diye T-TR01–10 olur.
T-TR11 export/player, T-TR12 demo/kapanış ek testleridir.

## Uygulama sırası

| Sıra | Görev | Ön koşul | Saat |
|---|---|---|---|
| 1 | [TR-01 — Trajectory sözleşmesi ve interpolasyon](../tasks/TR-01.md) | S3-02 | 12–18 |
| 2 | [TR-02 — Limitler ve otomatik süre ölçekleme](../tasks/TR-02.md) | TR-01 | 18–28 |
| 3 | [TR-03 — Waypoint, FK ve playback/export](../tasks/TR-03.md) | TR-02 | 12–20 |
| 4 | [TR-04 — Trajectory demo ve G-TR kabulü](../tasks/TR-04.md) | TR-03 | 10–16 |

## Görev bazında kabul

### TR-01 — Trajectory sözleşmesi ve interpolasyon

IK joint çıktısından bağımsız zaman-parametreli hareket üretmek.

**Gereksinim:** REQ-TR01. **Test:** T-TR01, T-TR02, T-TR03, T-TR07.

**Çıktı:** Trajectory: timestamps [T], q/dq/ddq [T,DOF], robot/revision, limit profili, yöntem, duration/timestep, kaynak hedef kimlikleri. Linear baseline ve varsayılan quintic; q0/qf, uçlarda dq=ddq=0. Linear baseline uç velocity sıçramasını açık etiketle. Analitik polinom/türev hesabı; monoton timestamps, zero-distance, bozuk/sonlu olmayan giriş ve invalid duration testleri. IK/FK/robot tanımını kopyalama; solver hedefi dışarıdan gelsin.

**Kabul:** T-TR01–03: endpoint ve hız/ivme sınır koşulları, analitik referansla ön kayıtlı sayısal toleransta PASS. T-TR07: zaman metadata alanları hariç aynı girdide aynı sayısal trajectory; test toleransları uygulama başlamadan sabit.

**Başarısızlıkta:** Bozuk zaman veya shape reddedilir; interpolasyon sonucu collision-free veya fiziksel uygulanabilir diye etiketlenmez.

**Kanıt:** `experiments/TR-01/`; PLANLANDI / NOT_RUN.

### TR-02 — Limitler ve otomatik süre ölçekleme

Joint konum/hız/ivme limitlerini süre hesabıyla korumak.

**Gereksinim:** REQ-TR02. **Test:** T-TR04, T-TR05, T-TR06.

**Çıktı:** Merkezi RobotModel ve ayrı sürümlenmiş motion profile kullan; URDFde olmayan acceleration limitini uydurma. Quintic için analitik türev ekstremumlarından veya kanıtlı üst sınırdan süre hesapla; yalnız örnek gridinin düşük maksimumunu gerçek maksimum sayma. Tüm jointler için ortak duration; yetersiz kullanıcılı süreyi açıkça reddet veya izinli deterministic scaling sonucu raporla. Eksik limit UNKNOWN_LIMIT; simülasyon için açık EXPERIMENTAL_LIMITS profili ayrı, physical-ready=false.

**Kabul:** T-TR04–06: tüm segmentlerde konum/hız/ivme sınırları doğrulanır; analitik ekstremum ve örneklem testleri uyumlu. Bir limit eksikken tam motion-limits PASS yok; yalnız belirtilen deneysel simülasyon profiliyle kapsamlı kabul olabilir.

**Başarısızlıkta:** Duration yükseltmek IK/çarpışma sorununu çözmez; time scaling sonrası tüm kontroller tekrar yapılır.

**Kanıt:** `experiments/TR-02/`; PLANLANDI / NOT_RUN.

### TR-03 — Waypoint, FK ve playback/export

current→A→B ve çoklu waypoint hareketini kaydedilebilir oynatmaya dönüştürmek.

**Gereksinim:** REQ-TR03. **Test:** T-TR08, T-TR09, T-TR11.

**Çıktı:** A/B hedeflerini ortak IK servisi çözer; generator hedef jointleri alır. İlk sürümde segment uçları sıfır hız/ivme ile durur. Her örnekte FK, A/B son pose ve joint limit kontrolü; bounded joint değerlerini wrap etme. Bağımsız player: load/play/pause/stop/seek/speed/current_state; playback hızını robotun fiziksel limit uygunluğu ile karıştırma. JSON ve CSV/NPZ export, schema/version/hash; gelecekte ROS mesaj adapterı, şimdi zorunlu ROS bağımlılığı yok.

**Kabul:** T-TR08/09: waypoint sırası ve endpoint FK, aynı IK profil toleranslarıyla doğru; segment sürekliliği kayıtlı. T-TR11: export-import sayısal eşliği ve seek/pause/stop; görünüm zamanı fiziksel yürütme emri değil.

**Başarısızlıkta:** Yarım veya yanlış robotlu trajectory yüklenmez; smoothing/blending ilk teslimata gizlice eklenmez.

**Kanıt:** `experiments/TR-03/`; PLANLANDI / NOT_RUN.

### TR-04 — Trajectory demo ve G-TR kabulü

Trajectory katmanını tekrar üretilebilir demo ve regresyonla kapatmak.

**Gereksinim:** REQ-TR04. **Test:** T-TR10, T-TR12.

**Çıktı:** KUKA current→A→B deterministik demo: duration, timestep, örnek sayısı, dq/ddq maxima, limit ihlali, A/B pose hata, solver/interpolator. q/dq/ddq–zaman, TCP XYZ ve 3B path grafikleri; ham trajectory ve config hashleri. F0/Core/H2/MR ilgili regresyonları, TR bütün testleri; MP için waypoint zamanlama API ve limit profili devri.

**Kabul:** T-TR10/12: demo endpoint/limit/türev/determinism testleri PASS ve grafikler ham sayılarla tutarlı. G-TR yalnız TR-01/02/03 kabulü ile kapanır; obstacle avoidance ve fiziksel hareket NOT_IMPLEMENTED olarak kalır.

**Başarısızlıkta:** Eksik limit veya validation hatasını demo animasyonuyla gizleme; MP başlamaz.

**Kanıt:** `experiments/TR-04/`; PLANLANDI / NOT_RUN.

## Faz kapısı

**G-TR:** endpoint, analitik hız/ivme, position/velocity/acceleration limitleri,
determinism, multi-waypoint, FK, regresyon ve demo PASS. Deneysel ivme profili
varsa yalnız o simülasyon kapsamı kabul edilir; fiziksel limit doğrulaması ayrı.
Sonraki iş MP-01. Sürekli collision-free veya physical-ready iddiası yok.

Tüm işler için gereksinim → değişiklik → test → ham kanıt bağı tutulur.
Uygulamada protokol/seed/kimlikler sonuç görülmeden dondurulur; başarısız kayıtlar
korunur. Yukarıdaki süreler tahmindir; GPU bekleme, donanım temini ve kullanıcı
testi takvimini kapsamaz. İlk iki görevden sonra yalnız tahminler güncellenir.

[Ana roadmap](MASTER_ROADMAP.md) · [Kaynak eşlemesi](SOURCE_MAPPING_r3.md)
· [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).
