# ROADMAP EXTENSION — Collision-aware motion planning

Roadmap r3 · 11 Ekim 2026 · Durum PLANLANDI / NOT_STARTED · Tahmin: 66–112 etkin saat.
Yazılım hedef sürümü belge revizyonundan ayrıdır; henüz release yapılmadı.

G-TR sonrası. Statik sahnede q_start→q_goal geometrik yol arar; çalışma
zamanında çıktı TRye gider. İlk planner RRT-Connect ve direct-edge kısa yolu;
state/edge collision, raw/simplified path, limit ve scene revision tek sözleşmede.
Visual ve collision mesh ayrıdır. Genel planner bağımsız API, ilk teslimatta
MoveIt/OMPL zorunlu bağımlılığı yoktur. Dinamik engel/perception/grasp/torque kapsam dışıdır.

## Uygulama sırası

| Sıra | Görev | Ön koşul | Saat |
|---|---|---|---|
| 1 | [MP-01 — Planning scene ve state/edge denetimi](../tasks/MP-01.md) | TR-04 | 12–20 |
| 2 | [MP-02 — Direct edge ve RRT-Connect](../tasks/MP-02.md) | MP-01 | 18–30 |
| 3 | [MP-03 — Path sadeleştirme ve trajectory entegrasyonu](../tasks/MP-03.md) | MP-02 | 12–20 |
| 4 | [MP-04 — Planning benchmark ve KUKA demoları](../tasks/MP-04.md) | MP-03 | 14–24 |
| 5 | [MP-05 — G-MP kabulü ve Studio devri](../tasks/MP-05.md) | MP-04 | 10–18 |

## Görev bazında kabul

### MP-01 — Planning scene ve state/edge denetimi

H2 geometri servisini GUI bağımsız statik planning scene ile birleştirmek.

**Gereksinim:** REQ-MP01. **Test:** T-MP02, T-MP03, T-MP05, T-MP06, T-MP07.

**Çıktı:** PlanningScene robot/revision/state, box/sphere/cylinder obstacle ID/pose/dimensions/enabled, allowed pairs ve workspace metadata. URDF collision mesh veya doğrulanmış sade geometri; visual render meshinden sessiz collision modeli türetme. Start/goal limit, environment ve self collision; qA→qB edge örnekleme çözünürlüğü ve kapsamı kaydet. H2 state/edge checker tekrar kullan; UNKNOWN geometri plan için valid sayılmaz; statik scene revisionu her çıktı ile eşleşsin.

**Kabul:** T-MP02/03: colliding start/goal erken hata; T-MP05/06: ara edge/joint limit ihlalleri yakalanır. T-MP07: desteklenen self collision çiftleri doğrulanır; kapsam dışı robotta açık UNKNOWN/reject. KUKA tesliminde self collision sessizce atlanamaz.

**Başarısızlıkta:** Sonlu edge sampling sürekli garanti değildir; sonucu SAMPLED_PATH_CHECK olarak etiketle.

**Kanıt:** `experiments/MP-01/`; PLANLANDI / NOT_RUN.

### MP-02 — Direct edge ve RRT-Connect

Geçerli q_start ve q_goal arasında engelden kaçan geometrik yol aramak.

**Gereksinim:** REQ-MP02. **Test:** T-MP01, T-MP04, T-MP08.

**Çıktı:** Önce direct edge; geçerliyse arama açma. Engelde ortak state/edge validator ile RRT-Connect. Planner bağımsız interface; random seed, sampling limits, bütçe, step/resolution, iptal configi. PlanningResult: path, status/failure_reason, elapsed, sampled states, collision checks, path length, planner/scene/robot kimliği ve seed. NO_PATH_FOUND/TIMEOUT çözüm yokluğunun ispatı değildir; Cartesian goal önce IKden gelir.

**Kabul:** T-MP01: boş sahnede direct path; T-MP04: bilinen mümkün engelli sahnede alternative path ve bağımsız validator. T-MP08: aynı seed ve sabit iş/iterasyon bütçesinde aynı sayısal sonuç; wall-clock timeout nondeterminism ayrıca raporlu.

**Başarısızlıkta:** RRT*/PRM/MoveIt veya dinamik engel kapsamını bu göreve katma; bütçe bittiğinde yapılandırılmış hata.

**Kanıt:** `experiments/MP-02/`; PLANLANDI / NOT_RUN.

### MP-03 — Path sadeleştirme ve trajectory entegrasyonu

Ham yolu geçerliliği koruyarak sadeleştirip TR katmanına vermek.

**Gereksinim:** REQ-MP03. **Test:** T-MP09, T-MP10, T-MP11.

**Çıktı:** Raw/simplified path ayrı; her shortcut edge yeniden collision kontrolü; seed/config ve önce/sonra maliyet sakla. TR generate_from_waypoints; time parameterization MP içinde yeniden yazılmasın. Retiming/smoothing varsa nihai interpolated trajectory tekrar limit/collision/FK denetimi; waypoint-only kontrol yetmez. Goal IK birden fazla branch arayüzüne açık olabilir; ilk sürüm tek kabul edilmiş hedef, planner-aware seçimi ayrı gelecek deney.

**Kabul:** T-MP09: shortcut geçerliliği ve endpoint koruma; T-MP10: TR limitleri korunmuş. T-MP11: final q ve FK hedefi solver toleransında; bütün denetim sonuçları aynı robot/scene revisionuna ait.

**Başarısızlıkta:** Sadeleşen yolun veya interpolasyonun validitysi bozulursa raw doğrulanmış yolu kullan veya açık hata; sorunlu sonucu yürütmeye geçirme.

**Kanıt:** `experiments/MP-03/`; PLANLANDI / NOT_RUN.

### MP-04 — Planning benchmark ve KUKA demoları

Planlayıcı için tek başarılı demo yerine tekrarlanabilir sahne dağılımı oluşturmak.

**Gereksinim:** REQ-MP04. **Test:** T-MP13.

**Çıktı:** Önerilen yeni deney tasarımı: easy/medium/hard sınıf başına 10 sabit sahne ve 10 seed =300 deneme. Protokol/timeout/edge resolution final öncesi dondurulur. Başarı, median/P95 süre, path length, raw/simplified waypoint, sampled states, collision checks ve mümkünse clearance; başarısız denemeler dahil. KUKA doğrudan yolun çarpıştığı engel arkası hedef ve current→A→B demo; raw/simplified/TCP trace/scene görselleri. PlanningService API: robot/scene/start/goal, solve/plan/validate/generate/result; GUI planner algoritmasına bağlanmaz.

**Kabul:** T-MP13: 300 denemenin tamamı raporlu, yanlış kabul 0; iki deterministic demo geçerli alternatif path üretir. Genel planner başarı/P95 hedefleri sahne protokolünde uygulamadan önce tanımlanır; bu roadmap ölçülmemiş evrensel hız/yüksek başarı yüzdesi ilan etmez.

**Başarısızlıkta:** Başarı hedefi sağlanmazsa GP açık; yalnız başarılı seedleri seçerek benchmark PASS üretme.

**Kanıt:** `experiments/MP-04/`; PLANLANDI / NOT_RUN.

### MP-05 — G-MP kabulü ve Studio devri

Motion planning katmanını destek profili ve regresyonlarla kapatmak.

**Gereksinim:** REQ-MP05. **Test:** T-MP12, T-MP14.

**Çıktı:** T-MP01–14 sonuçları; F0/Core/H2/MR/TR ilgili regresyonları; temiz demo replay. Yalnız statik scene, desteklenen collision geometri ve sampled-path kapsamını paketle. Future planner/OMPL adapter noktası bırak; şimdi MoveIt bağımlılığı ekleme; Studio çizim/veri sözleşmesi ve hataları devret.

**Kabul:** T-MP12/14: kritik regression yok, kaynak belgenin start/goal/edge/limits/planner/simplification/TR/finalpose/demo koşulları PASS. G-MP geçmeden S3-03 birleşik planning/playback deneyimi ve PE geliştirilmeye başlanmaz.

**Başarısızlıkta:** Fiziksel safe motion iddiası yok; planlamanın başarısı collision-free simulation path kapsamındadır.

**Kanıt:** `experiments/MP-05/`; PLANLANDI / NOT_RUN.

## Faz kapısı

**G-MP:** kaynak T-MP01–12 ve yeni benchmark/kapanış testleri PASS;
state/edge/limits/shortcut/TR/finalpose valid, iki KUKA demo ve ölçümlü benchmark.
Sonlu collision örneklemesi sürekli garanti değildir; SAMPLED_PATH_CHECK kapsamı
korunur. Sonraki ürün işi S3-03; fiziksel uzantı PE-07A ayrıca bu kapıya bağlıdır.

Tüm işler için gereksinim → değişiklik → test → ham kanıt bağı tutulur.
Uygulamada protokol/seed/kimlikler sonuç görülmeden dondurulur; başarısız kayıtlar
korunur. Yukarıdaki süreler tahmindir; GPU bekleme, donanım temini ve kullanıcı
testi takvimini kapsamaz. İlk iki görevden sonra yalnız tahminler güncellenir.

[Ana roadmap](MASTER_ROADMAP.md) · [Kaynak eşlemesi](SOURCE_MAPPING_r3.md)
· [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).
