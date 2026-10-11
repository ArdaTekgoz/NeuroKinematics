# Dört yeni kaynak → görev → test eşlemesi

11 Ekim 2026 · Plan r3 · Dört PDFnin toplam **191 sayfası** metin olarak
incelendi; seçili sayfalar ayrıca görüntü üzerinden kontrol edildi. Belgelere
uygulama sonucu muamelesi yapılmadı. Orijinaller yerinde, değiştirilmeden korunur.
“Yerleştirme” roadmap bağımlılıklarına yapılmıştır; PDFler taşınmadı/silinmedi.

| ID | Kaynak | Sayfa | SHA-256 |
|---|---|---|---|
| 1 | [Motion planning_261008_141729.pdf](<../../Eklenilecek yeni aşamalar/Motion planning_261008_141729.pdf>) | 44 | `e89621523dc7b3e34d9d37ac65438e3d768a5070705a252fe0c15bd179eb665f` |
| 2 | [Multi robot generalization_261008_141846.pdf](<../../Eklenilecek yeni aşamalar/Multi robot generalization_261008_141846.pdf>) | 57 | `201b433432cb6d606d086c3e6dd0613001356b904bfcae365cbf28856193b98d` |
| 3 | [Roadmap extensions_261008_143912.pdf](<../../Eklenilecek yeni aşamalar/Roadmap extensions_261008_143912.pdf>) | 62 | `721f3c5d93446d5fe93ecbe51300ded340c8a9331e95041c43ce0e08c4832622` |
| 4 | [Trajectory Katmanı_261008_141531.pdf](<../../Eklenilecek yeni aşamalar/Trajectory Katmanı_261008_141531.pdf>) | 28 | `8b5438e8dbbd36cd5d83b36dcb5ae341096b6c8393f276dc11c87a5078bd0b9f` |

## Kabul edilen ve uyarlanan kapsam

| Kaynak / sayfa | Gereksinim grubu | Uygulama yeri / test | Karar ve gerekçe |
|---|---|---|---|
| MR 1–8 | Hybrid sonrası, Studio öncesi generic mimari | MR-01–06; G-MR | Kabul; S3-01 H3 silinmez, MR kanıtını değerlendirir |
| MR 8–14 | RobotModel, URDF, zincir/ad/limit/frame doğrulama | MR-01; T-MR01–05 | Kabul; ilk destek 6DOF seri revolute, destek dışına açık hata |
| MR 14–20 | FK/Jacobian/numerical-ready, sampling/workspace | MR-02/03; T-MR06–09 | Kabul; her robot bağımsız doğrulama ve ön kayıtlı numerical benchmark ister |
| MR 20–25 | Ortak eğitim/hybrid onboarding | MR-03/04/05; T-MR10/11 | Kabul; per-robot eğitim, zero-shot değil |
| MR 25–30 | Readiness, API, checkpoint/restart | MR-04/06; T-MR13/16 | Tamamlandı ile kalite geçti ayrılır; süre garantisi yok |
| MR 31–35 | Numerical/Fully ready, üç robot, KUKA regresyon | MR-05; T-MR11/14/15 | Kabul; FULLY_READY için neural/hybrid kalite gate, salt training complete değil |
| MR 36–42 | Paket/manifest/registry/revision invalidation | MR-04; T-MR12/13 | Kabul; URDF/TCP değişiminde bağlı artifactlar stale |
| MR 43–45, 49–52 | Generic API ve KUKA bağımlılığı yasağı | MR-02/06, S3-02; T-MR15/16 | Uyarlama: ortak motor KUKA bilmez; kanıtlı analitik plugin exact-hash capability ile korunur |
| MR 46–48, 51–57 | T-MR01–15 ve nihai onboarding kabulü | MR-01–06 | Orijinal test kimlikleri korundu; T-MR16 temiz API devri ek |
| MR 45–46 | MAML/arbitrary DOF/universal NN vb. | RESEARCH_BACKLOG | Ertelendi; generic pipeline, evrensel ağ iddiası değildir |
| TR 1–6 | Katman sınırı, trajectory şeması, linear baseline | TR-01; T-TR01/07 | Kabul; GUIden bağımsız servis |
| TR 6–10 | Quintic, q/dq/ddq, limits, duration, waypoints | TR-01/02/03; T-TR01–08 | Kabul; örnekler dışında analitik ekstremum denetimi eklendi |
| TR 10–15 | IK/FK, player, export, ortak model | TR-03; T-TR09/11 | Kabul; IK yeniden yazılmaz, ROS zorunlu değil |
| TR 15–19 | T-S01–10 | T-TR01–10 | Yeniden adlandırıldı; mevcut Studio T-S01–07 kimlikleriyle çakışma önlendi |
| TR 19–28 | A/B demo, grafikler, regression, PASS | TR-04; T-TR10/12 | Kabul; gerçek hareket ve obstacle avoidance ayrı aşama |
| MP 1–8 | TR sonrası geliştirme, TR öncesi runtime | MP-01–05 | Kabul; Studio S3-02 kabuğu ön koşul, S3-03 birleşik akış sonrasında |
| MP 8–13 | Scene/geometri/self/edge collision | H2-04 → MP-01; T-H05, T-MP02/03/05/06/07 | H2deki denetleyici tekrar kullanılacak; “mümkünse self” sessiz güvenlik varsayımı olamaz |
| MP 13–18 | C-space, RRT-Connect, result, seed, direct edge | MP-02; T-MP01/04/08 | Kabul; sabit iş bütçesinde determinism ile wall-clock timeout ayrıldı |
| MP 18–23 | Shortcut, metrics, trajectory, IK integration | MP-03; T-MP09–11 | Kabul; nihai interpolasyon da yeniden kontrol edilir |
| MP 23–29 | KUKA iki demo, service, görseller, benchmark | MP-04; T-MP13 | Kabul; r3te önerilen 30 sahne×10 seed protokolü eklendi |
| MP 30–35, 40–44 | T-MP01–12, plugin, kapanış | MP-05; T-MP12/14 | Kabul; kaynak testler korunur; MoveIt/OMPL opsiyonel adapter |
| MP 35–39 | Kapsam dışı ve safety terminolojisi | MP roadmap / PE roadmap | Statik sahne ve sampled path; dynamic/perception/RL/grasp bu teslimat dışında |
| PE 1–6, 48–57 | Fazların korunması ve runtime katmanları | MASTER, ADR-027 | Kabul; yeni uzantılar eski F0/C1 kanıtını yeniden tanımlamaz |
| PE 6–12 | DT-REQ-01–05 | PE-01; T-PE01 | Uyarlama: twin VALID yazılım ön koşulu, tek başına fiziksel execution yetkisi değil |
| PE 12–18 | HW-REQ-01–05, ROS 2/ros2_control | PE-02; T-PE02; gerçek I/O PE-03/06 | Kabul; fake hardware ile gerçek hardware testinin durumu ayrı |
| PE 18–25 | EMB-REQ-01–06, controller/transport | PE-03; T-PE03 | Kabul; STM32/CAN örnek platform, corea zorunlu bağımlılık değil |
| PE 25–29 | Closed-loop actual/commanded tracking | PE-04; T-PE04 | Korundu; öncesine temel calibration/HIL/interlock alındı |
| PE 30–33 | Calibration/sim-to-real/revision | PE-05A/B; T-PE05A/B | Temel commissioning kalibrasyonu önce, held-out fiziksel karşılaştırma sonra |
| PE 34–37 | HIL fault injection | PE-06; T-PE06 | PE-04 öncesine alındı; hata yönetimi hareketten sonra yapılmaz |
| PE 37–39 | Physical safety architecture | PE-07A/B; T-PE07A/B | Gereksinimler başlangıca, interlock kabulü full movement öncesine |
| PE 39–44 | STUDIO-EXT | STUDIO-EXT; T-SX04 | Mevcut GUI uzantısı; driver/firmware GUI içinde değil |
| PE 44–48 | Plugin/SDK/reliability/productization | PE-08; T-PE08 | Kabul; endurance/deployment eşikleri platform netleşince ön kayıtlanır |
| PE 54–55 | Önerilen doğrusal kabul sırası | ADR-027, PE roadmap | Sıra bilinçli değişti: PE-07A→01→02→03→05A→06→07B→04→05B→STUDIO-EXT→08 |
| PE 58–62 | TK56 referansı, gelecekteki ürün hedefi, görev specification | PE-07A/08 ve tüm PE görevleri | TK56 aday; mevcut fiziksel özellikler varsayılmadı; task başına spec şartı |

## H2 eklerinin kaynağı ve kapsamı

H2-N1/N2 dört PDFnin içindeki bir yeni IK mimarisi talimatı değildir; kullanıcının
neural vizyonu ve C1 denetimindeki eksiklerin gerektirdiği ek işlerdir. REQ-HN01/02,
T-HN01/02 bu revizyonda tanımlanır. REQ-HG01/02/03/04/06 ve T-HG eşleri
geometrik motorun H2ye doğru alınmasını kapsar. Özgün REQ-H01–06, T-H01–07
değişmez. H1 P95/başarı/CI, A/B, ONNX, rollout ve collision adetleri korunur.

## Çakışma ve erteleme kaydı

- PDFdeki “her solver generic” isteği analitik KUKA bileşenini silmek şeklinde
  uygulanmadı: ortak servis generic, özel solver capability kontrollü plugindir.
- TRnin velocity/acceleration limitleri URDFde yoksa icat edilmez; ayrı sürümlü
  simülasyon profili açıkça deneysel, physical-ready değildir.
- “Pathin tamamı collision-free” ifadesi sonlu örneklenmiş yol kontrolünün
  doğrulanmış kapsamı ile raporlanır; sürekli matematiksel garanti çıkarılmaz.
- RRT*/PRM/BIT*, planner-aware IK, velocity blending, dynamic obstacles,
  perception/SLAM, RL/torque/grasp/task planning, MAML/GNN/universal NN,
  Jetson/TensorRT ve cloud özellikleri bugünkü zorunlu kapsama alınmadı.
  [Araştırma birikimi](../RESEARCH_BACKLOG.md) ve gelecekteki ayrı ADRlere bağlıdır.
- Yeni PDFler H2 toleranslarını veya C1 negatif sonuçlarını geriye dönük değiştirmez.
