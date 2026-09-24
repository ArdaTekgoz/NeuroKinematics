# C1-01 Aşama 1 inceleme ve dondurma

Tarih: 24 Eylül 2026 · Karar: **IN_PROGRESS / STAGE_1_COMPLETE** · T-C00: **NOT_RUN**

## Başlangıç kapısı

Başlangıç `main` / `e1971bf8b70154d6f4d882546c20de0eaab2d83d`; `origin/main` aynı SHA. F0-06 `COMPLETE`, G0 `PASS / ACCEPTED`, Core `READY / NOT_STARTED` belgeleri okundu. `handoff-inputs.json` içindeki 15/15 dosya mevcut ve SHA-256 eşleşti. F0-05 sorgu listesi 12.000 kayıt manifest hash'iyle eşleşti. Robot `kuka_kr6_r900_sixx`, `base_link`→`tool0`, altı joint sırası, metre/radyan ve `wxyz` config doğrulayıcısıyla sınandı. Çalışma ağacında önceden var olan iki ilgisiz, izlenmeyen Word dosyası korundu; commit kapsamına alınmayacak.

## Platform kararı ve seçenekler

| Seçenek | Karar ve gerekçe |
|---|---|
| Tüm yöntemler native Windows | Seçilmedi. Bu makinede ROS 2/MoveIt, WSL dağıtımı veya Docker yok; dört plugin için Windows ortak kurulum kanıtı yok. |
| Tek Linux/ROS 2 ortamı | **Seçildi:** Ubuntu 24.04 LTS x86_64, ROS 2 Jazzy, MoveIt 2.15.2. Tüm yöntemler aynı hostta yeniden çalıştırılacak. Temiz kurulum lock'u Stage 2 smoke kapısı. |
| İzole subprocess worker | **Seçildi:** yöntem başına kalıcı yerel worker; aynı request/reply protokolü ve dış uçtan uca ölçüm. C++ pluginler ve Python DLS ayrıdır, worker/IPC maliyeti ana süreye dahildir. |
| Yalnız fonksiyonel entegrasyon | Fallback: aynı host ve measurement layer sağlanamazsa fonksiyonel kanıt ayrı sunulur, gecikme sıralaması `NOT_COMPARABLE` kalır; T-C00 kapanmaz. |

Kanonik Windows F0-05 DLS süresi Linux yöntemleriyle hız sıralamasına giremez. Linux'ta DLS yeniden çalıştırılır. CPU seti, governor, thread sayısı ve runtime sürümleri gerçek yürütmede kaydedilir. Solver iç zamanı tanısaldır; ana metrik çağrı, IPC, dönüşüm, bağımsız doğrulama dahil toplam süredir. Worker sıcak başlatılır; yükleme ve her deadline öncesi 20 sorgu warm-up ayrı ölçülür.

## Solver fizibilitesi

| Varyant | URDF, seed, limit ve timeout | Iteration / rastgelelik | Durum |
|---|---|---|---|
| DLS/default | Mevcut RobotSpec ve `q_current`; F0-05 10/50 ms; Pinocchio final kontrol | Iteration mevcut; restarts=0 | F0-05 tarihsel PASS; Linux ortak koşu `NOT_RUN` |
| KDL/default | MoveIt RobotModel, seri `base_link`→`tool0` chain; `ik_seed_state=q_current`; URDF limitleri; `searchPositionIK(timeout)` | Plugin API iteration vermez; `NOT_AVAILABLE`. İlk deneme seed, sonra random reseed olası; outer cap uygulanır. | Entegrasyon `NOT_RUN` |
| TRAC-IK/speed | Aynı RobotModel/chain; seed vektörü; URDF limitleri; çağrı timeout; epsilon `1e-5` | İki iç çözüm yolu/thread, random jumps; iteration API `NOT_AVAILABLE` | Entegrasyon `NOT_RUN` |
| pick_ik/local | Aynı MoveIt chain, seed ve limitler; `mode=local`, çağrı timeout | Gradient descent; iteration API `NOT_AVAILABLE`; tekrar edilebilirlik smoke ile doğrulanacak | Entegrasyon `NOT_RUN` |
| pick_ik/global | Ayrı `mode=global`; evrimsel arama, `memetic_num_threads=2`; aynı seed/limit/timeout | Stochastic; explicit seed hook doğrulanamazsa varyans kaydı ve determinism `NOT_AVAILABLE` | Entegrasyon `NOT_RUN` |

URDF chain doğrudan tüketilebilirlik kaynak API'lerden **öngörülen**, bu robotla çalışan plugin testi **NOT_RUN**. MoveIt group `base_link`→`tool0`, joint isim sırası ve sıfır olmayan TCP rotasyonu smoke sırasında bağımsız FK ile sınanacak. `wxyz`↔ROS `xyzw` dönüşümü yalnız adapter sınırında, roundtrip testi zorunlu. Yanlış eklem sırası veya limitler reddedilir. `q_target` worker isteğine hiç konmaz.

KDL `epsilon=1e-5`, `max_solver_iterations=500`, orientation weight=1; TRAC-IK `epsilon=1e-5`, Speed modu; pick_ik iç position/orientation eşikleri 1 mm/0,5 derece olarak donduruldu. Solverın iç `SUCCESS` sonucu ortak başarı değildir. Beş varyantın tamamı aynı Profile A (2 mm/1°), B (1 mm/0,5°), 10/50 ms, 5 geçiş ve aynı 12.000 sorguda bağımsız Pinocchio kontrolüne tabidir. Collision her zaman `NOT_CHECKED`.

## Ortak adapter ve veri sözleşmesi

Normatif alanlar [baseline-config.json](baseline-config.json) içindedir. Girdi `query_id`, `q_current`, hedef pozisyon ve `wxyz`, base/TCP, joint sıra/limit, mutlak deadline, config hash ve varsa seed. Çıktı solver kimliği/varyant/sürüm, native ve normalize durum, aday/null, termination, timeout, iteration/`NOT_AVAILABLE`, solver iç/IPC/doğrulama/toplam süre, bağımsız A/B ve limit sonucu, collision, hata sınıfı, log bağı. Durumlar timeout, çözümsüz, matematiksel geçersiz çıktı, limit, kurulum, process, adapter ve doğrulama hatalarını ayırır. Bir solver başarısızlığı erişilemezlik kanıtı değildir; `PROVEN_UNREACHABLE` ancak bağımsız analitik sertifikayla.

F0-05 `benchmark-schema.json` DLS `const` alanları ve DLS'e özel iteration/200 cap içerir. Eski dosya ve hash değişmez. Stage 2 yeni sürümlü C1-01 sonuç şeması ile solver registry/variant kuracak, eski F0-05 doğrulayıcısını regresyon olarak koruyacak. Beş varyantın ham satırları aynı `query_id` ve frozen query-list SHA ile bağlanacak; hiçbir başarısız satır atılmayacak. Git dışı tam JSONL için hash, bayt, satır, üretim komutu ve saklama durumu zorunlu.

## Planlanan T-C00 matrisi

1. Worker/plugin unit; pose/frame/quaternion roundtrip; yanlış sıra/limit; NaN/Inf/boyut/eksik alan.
2. Timeout/late reply, process crash, kurulum eksikliği, native→common status, iteration bulunmama.
3. Her varyantta kolay, sınır ve tekillik smoke; aynı Pinocchio doğrulaması; mutasyon/negatif testler.
4. F0-00–F0-06 regresyonları; küçük uçtan uca smoke. Başarısız smoke halinde tam benchmark `NOT_RUN`.
5. Smoke geçerse frozen 12.000 sorgu × 2 deadline × 5 pass = **120.000 deneme/varyant**, beş varyant için 600.000 satır hedefi. Bu sayı **plan**, ölçüm değil.
6. Subset/local-wide/profile kırılımları; bütün ve başarılı sorgu P50/P95/P99; başarı/hata/timeout/limit dağılımı; native/internal ve toplam süre ayrı.

## Risk, fallback ve emek kapısı

| Risk | Kapı / fallback |
|---|---|
| Linux ortamı bu Windows hostunda yok | Stage 2'de aynı fiziksel hostta WSL2 veya yerel Linux sağlanmalı; başka makinede ölçüm olursa bütün yöntemler o makinede yeniden koşmalı. |
| ROS deb transitif sürümler bu aşamada kilitlenemiyor | Smoke öncesi exact package/version, apt kaynak snapshot, compiler/runtime ve SHA kaydı; tutarsızlıkta tam benchmark yok. |
| pick_ik deprecated; stochastic global davranış | Kaynak değiştirilmez; local/global ayrı config; seed hook yoksa `NOT_AVAILABLE`, pass varyansı raporlanır. |
| 10 ms IPC + doğrulama maliyeti yüksek | Toplam süre saklanır; iç solver zamanı ana süre yerine geçmez; deadline başarısı düşükse dürüstçe raporlanır. |
| Worker/plugin URDF joint/TCP uyumsuzluğu | Kolay smoke ve FK roundtrip aşamasında dur; ortak robot sözleşmesi değiştirilmez. |
| DLS proje lisansı beyan edilmemiş | Harici yeniden dağıtım yapılmaz; lisans kaydı Stage 2 öncesi netleştirilir. |
| 20 saat etkin emek tavanı | Oturum bazında elle başlama/bitiş etkin süre ve bekleme ayrı yazılır; şu an etkin emek `NOT_MEASURED`. 20 saat dolarsa `BLOCKED`/`PARTIAL`, eksik solver gizlenmez. |

## Aşama 2 giriş şartı

Açık kullanıcı onayı, bu Stage 1 commit'inin remote'da olması ve hash eşleşmesi, Linux/Jazzy ve ROS bağımlılık lock'u, robot/plugin kolay smoke, ortak ölçüm ve validator doğrulaması. Bu inceleme solver kurulumu, entegrasyon veya tam benchmark içermez. G0 tarihsel kanıtları değişmedi; Core aktif faz kaydı yalnız kapı sonrası güncellendi.
