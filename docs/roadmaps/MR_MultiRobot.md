# ROADMAP EXTENSION — Generic robot ve otomatik onboarding

Roadmap r3 · 11 Ekim 2026 · Durum PLANLANDI / NOT_STARTED · Tahmin: 88–152 etkin saat.
Yazılım hedef sürümü belge revizyonundan ayrıdır; henüz release yapılmadı.

Hybrid/G2 sonrası, Studio öncesi. İlk destek fixed-base açık zincir altı
revolute eklem/tek uçtur. Bu aşama evrensel zero-shot neural IK değildir.
KUKA doğrulanmış referans, ikinci üretici robot ve custom test robotu gereklidir.
KUKA analitik motoru açık capability plugin olarak korunur; generic numerical
motorlar yeni robot için özel kod gerektirmez. Studio doğrudan somut robota bağlanmaz.

## Uygulama sırası

| Sıra | Görev | Ön koşul | Saat |
|---|---|---|---|
| 1 | [MR-01 — Generic robot sözleşmesi ve import](../tasks/MR-01.md) | H2-06 | 10–16 |
| 2 | [MR-02 — Generic FK/Jacobian ve numerical-ready](../tasks/MR-02.md) | MR-01 | 14–24 |
| 3 | [MR-03 — Robot bağımsız veri ve eğitim pipeline](../tasks/MR-03.md) | MR-02 | 16–28 |
| 4 | [MR-04 — Registry ve kesintiye dayanıklı onboarding](../tasks/MR-04.md) | MR-03 | 16–28 |
| 5 | [MR-05 — Üç robotta uçtan uca kabul](../tasks/MR-05.md) | MR-04 | 20–36 |
| 6 | [MR-06 — G-MR kapanışı ve Studio API devri](../tasks/MR-06.md) | MR-05 | 12–20 |

## Görev bazında kabul

### MR-01 — Generic robot sözleşmesi ve import

Studio öncesinde tek robot veri kaynağı ve doğrulanmış URDF girişini kurmak.

**Gereksinim:** REQ-MR01. **Test:** T-MR01, T-MR02, T-MR03, T-MR04, T-MR05.

**Çıktı:** RobotModel: joint/link/sıra, base/TCP, tür/axis/origin, position/velocity/acceleration kaynakları, visual/collision geometry ve revision. İlk destek fixed-base açık zincir, altı kontrollü revolute joint, tek uç; unsupported DOF/tür/mimic/kapalı zinciri açık reddet. KUKA, başka üreticinin 6DOF modeli ve custom 6DOF test modeli için varlık/lisans/kaynak commit pinleri; TK56 modeli hazır ve doğrulanmış değilse hazır sayma. Eksik/duplicate ad, zincir kopukluğu, limit, birim/frame ve mesh hataları için yapılandırılmış rapor.

**Kabul:** T-MR01–05: üç robot import; bozuk URDF ve eksik zorunlu limit reddi; hiçbir robotta sessiz KUKA defaultu yok. Model kimliği, base/TCP ve joint sırası tek kaynaktan türetiliyor; acceleration eksikliği açık alan.

**Başarısızlıkta:** Destek dışı robot INVALID; geniş DOF/robot ailesi desteği ayrı ADR ister.

**Kanıt:** `experiments/MR-01/`; PLANLANDI / NOT_RUN.

### MR-02 — Generic FK/Jacobian ve numerical-ready

Yeni robotu neural eğitimden önce doğrulanmış numerical hizmete almak.

**Gereksinim:** REQ-MR02. **Test:** T-MR06, T-MR07, T-MR08, T-MR15.

**Çıktı:** Generic FK/Jacobian backend ve bağımsız XML/geometrik/finite-difference karşılaştırması; yeni robota özel solver sınıfı yazma. KUKA referansını değiştirmeden aynı inputlarda generic sonuçla karşılaştır; DLS/LM robot sözleşmesi adapterları. Robot başına benchmark profili, tolerans, bütçe ve örnek sayısını sonuç görülmeden preregister et; F0 matematik eşiklerini gerekçesiz gevşetme. Analitik KUKA pluginini yalnız exact capability/hash eşleşmesinde sun; genel sayısal akışta KUKA isim/limit/sabitleri olmasın.

**Kabul:** T-MR06/07: her üç robotta bağımsız FK/Jacobian testleri ve hata enjeksiyonları PASS; KUKA regresyonu PASS. T-MR08: ön kayıtlı numerical IK kabul profili PASS olmadan NUMERICAL_READY verilmez. T-MR15: solver çekirdeğinde robot adına özel dal yok; plugin içindeki açık robot geometrisi izinli ve izole.

**Başarısızlıkta:** Import PASS tek başına ready değil; erişilemeyen ve çözülmemiş sorgu ayrımını koru. İkinci robot başarısı ölçülmeden genellik iddiası yok.

**Kanıt:** `experiments/MR-02/`; PLANLANDI / NOT_RUN.

### MR-03 — Robot bağımsız veri ve eğitim pipeline

H2 yeni eğitim hattını yeni robotlara config/veri üzerinden taşımak.

**Gereksinim:** REQ-MR03. **Test:** T-MR09, T-MR10.

**Çıktı:** Uniform/boundary/singularity-oriented veri, split ve train-only normalizasyon; KUKA sabitlerini factory dışına çıkar. Workspace poz/yön dağılımı, joint kullanımı ve manipulability raporları; veri boşlukları ve unreachable yorum sınırı. Ortak neural mimari ve training config: robot başına data/model registry; geometric teacher yokken doğrulanmış numeric teacher ve eksik etiket maskesi. Yeni robotta per-robot eğitim, üç seed/validation/freeze politikası; KUKA checkpointini zero-shot başarılı sayma.

**Kabul:** T-MR09: iki temiz üretim aynı örnek/hash, çapraz split sızıntı 0; üç robotta manifest var. T-MR10: model sınıfı/solver kodu değişmeden configten eğitim ve validation çalışır; performans ayrı raporlanır.

**Başarısızlıkta:** Eğitim tamamlandı readiness değildir; kalite gate geçmezse NUMERICAL_READY korunur, neural state NOT_MET.

**Kanıt:** `experiments/MR-03/`; PLANLANDI / NOT_RUN.

### MR-04 — Registry ve kesintiye dayanıklı onboarding

Hazırlık durumlarını, robot/model uyumluluğunu ve uzun görevleri kalıcı yönetmek.

**Gereksinim:** REQ-MR04. **Test:** T-MR12, T-MR13.

**Çıktı:** IMPORT→VALIDATE→NUMERICAL_READY→DATA→TRAIN→NEURAL_VALIDATE→HYBRID_VALIDATE akışı, her aşama için hashli kalıcı çıktı. IMPORTED/INVALID/NUMERICAL_READY/TRAINING/NEURAL_VALIDATED/HYBRID_VALIDATED/FULLY_READY/FAILED ayrı; completion ve quality ayrı alanlar. URDF/TCP/limit/normalizasyon değişince validation/dataset/model/trajectory dependent artifactlarını stale yap. add/prepare/status/metrics/list/delete API; checkpoint resume, iptal, atomik yazım, crash recovery; GUI bağımsızlığı.

**Kabul:** T-MR12: robot/model hash mismatch inference ve FULLY_READY durumunu engeller; eski ürün sessizce kullanılmaz. T-MR13: her uzun aşamada kesinti/yeniden başlatma; yalnız geçerli tamamlanmış çıktılar yeniden kullanılır; yarım çıktı başarıya dönüşmez.

**Başarısızlıkta:** Kesin hazır olma süresi sözü verme; bozuk checkpointte kontrollü yeniden başlangıç ve açık veri kaybı raporu.

**Kanıt:** `experiments/MR-04/`; PLANLANDI / NOT_RUN.

### MR-05 — Üç robotta uçtan uca kabul

KUKA + farklı üretici + custom robotta yeniden kod yazmadan onboarding kanıtı oluşturmak.

**Gereksinim:** REQ-MR05. **Test:** T-MR11, T-MR14, T-MR15.

**Çıktı:** Her robot için import/FK/Jacobian/numeric/data/neural/hybrid/runtime matrisi; benchmark eşikleri MR-02 ön kaydından. Geometrik plugin bulunan ve bulunmayan robotu aynı HybridSolver API ile çalıştır. KUKA F0/Core/H2 regresyonlarını yeniden çalıştır; performans değişimini aynı protokolle raporla. FULLY_READY için numerical, veri, neural kalite ve hybrid kabulü; salt training complete yeterli değil.

**Kabul:** T-MR11/14/15: üç robot aynı kodla çalışır; KUKA kritik regresyon yok; hashli sonuç ve kaynak sayımı hazır. G-MR için üç robot onboarding ve kaynak PDF tam kabul matrisi kapanmalı; başarısız neural kaliteye FULLY_READY verilmez. Yeni robot kalite eşikleri henüz ön kayıtlı değilse MR gate BLOCKED_SPEC olarak açık kalır; sonuç görüldükten sonra eşik seçilmez.

**Başarısızlıkta:** Yalnız KUKA geçerse generic faz PASS değil. Sayısal kullanım açık tutulabilir, Studio genel release gate kapatılamaz.

**Kanıt:** `experiments/MR-05/`; PLANLANDI / NOT_RUN.

### MR-06 — G-MR kapanışı ve Studio API devri

Generic robot paketini ve Studio için değişmez servis sözleşmesini teslim etmek.

**Gereksinim:** REQ-MR06. **Test:** T-MR16.

**Çıktı:** Temiz kilitli ortamda custom URDF import→ready replay; model/config/artifact manifestleri ve rehber. Robot registry, solver capability, onboarding progress/fault API örnekleri; tüm hazır durumların kapsam açıklaması. S3-01 H3 için robot B pipeline kimliği, code-change sayımı ve kıyas verilerini devret; altyapıyı yeniden yazdırma.

**Kabul:** T-MR16: temiz replay, bozuk paket reddi, üç robot readiness kanıtı ve G-MR kararı mevcut. GUI hiçbir somut KukaRobot implementationına bağlanmadan API ile kullanılabilir.

**Başarısızlıkta:** Genel API veya çoklu robot kabulü eksikse S3-02 açılmaz; partial sonucu tam generic support gibi sunma.

**Kanıt:** `experiments/MR-06/`; PLANLANDI / NOT_RUN.

## Faz kapısı

**G-MR:** kaynak PDFdeki üç robot kabulü, numerical-ready, veri/eğitim hattı,
hybrid uyumu, model/hash/restart ve KUKA regresyonu kapanmış olmalı.
FULLY_READY yalnız eğitim bitti demek değildir; robot bazlı ön kayıtlı kalite
kapıları gerekir. Eksik neural kalite NUMERICAL_READY kullanılmasını engellemez,
ancak generic fazın tam kabulünün yerine geçmez. Sonraki iş S3-01.

Tüm işler için gereksinim → değişiklik → test → ham kanıt bağı tutulur.
Uygulamada protokol/seed/kimlikler sonuç görülmeden dondurulur; başarısız kayıtlar
korunur. Yukarıdaki süreler tahmindir; GPU bekleme, donanım temini ve kullanıcı
testi takvimini kapsamaz. İlk iki görevden sonra yalnız tahminler güncellenir.

[Ana roadmap](MASTER_ROADMAP.md) · [Kaynak eşlemesi](SOURCE_MAPPING_r3.md)
· [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).
