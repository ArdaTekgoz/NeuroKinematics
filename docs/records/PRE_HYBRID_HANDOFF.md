# Hybrid öncesi proje devri

11 Ekim 2026 · Belge r1 · Ana plan r3 · Çalışma branch'i `codex/c1-06r`.
İlk okuma [START_HERE](../../START_HERE.md); görev yönetimi [AGENTS](../../AGENTS.md).

## Mevcut karar ve açık hedefler

| Konu | Gerçek durum | Kaynak |
|---|---|---|
| Foundations/G0 | PASS / ACCEPTED; bağımsız FK/Jacobian ve deterministik veri | [G0](../../experiments/F0-06/G0_DECISION.md) |
| Core/G1 | Araştırma kapanışı PASS; direct neural ürün NOT_MET | [G1](../../experiments/C1-07/G1_DECISION.md) |
| C1-06S | Robot-özel geometrik IK PASS; learned=false; yeni final 12.000/12.000 A/B | [Kabul](../../experiments/C1-06S/acceptance.json) |
| Neural eğitim | Yeni model eğitilmedi; eski 6 aday araştırma amaçlı | [Neural devir](../../experiments/C1-07/HYBRID_HANDOFF.json) |
| Neural katkı H1 | NOT_MEASURED; yeni eğitim H2-N1/N2, eşli deney H2-03 | [H2 şartname](../roadmaps/H2_REVISION_SPEC.md) |
| Hybrid | NOT_STARTED; G2 ve G-N açık | [H2 planı](../roadmaps/H2_Hybrid.md) |
| Collision/ONNX/generic robot/fiziksel deney | NOT_RUN; ilerideki görevler | [Roadmap r3](../roadmaps/MASTER_ROADMAP.md) |

Core araştırmasındaki reddedilen **H2 hipotezi**, Hybrid fazı **H2-01** görevinden
farklıdır. Başarı sıfır C1-04/05 validation'da zaten görülmüştür; C1-06'da aniden
oluşmamıştır. Decoder/metadata kusurları düzeltilmiş, genelleşme sorunu kalmıştır.
[Kök neden](../../experiments/C1-06S/ROOT_CAUSE_AND_RECOVERY.md) ve
[hazırlık denetimi](../../experiments/PLAN-20261011-ROADMAP/C1_READINESS_AUDIT.md) okunmalıdır.

## Çözücülerin görevleri

`AnalyticIK.solve(q_current, target_position_m, target_quaternion_wxyz)` exact
KR6 robot/TCP için cebirsel aday üretip limit ve bağımsız FK denetimi yapar.
H2'de kimliği doğrulanmış ayrı motor olarak değerlendirilecek; yeni ağ veya
robotlar arası genel çözücü değildir. [Geometrik devir](../../experiments/C1-06S/HANDOFF.json)
robot, eklem sırası, limitler ve kaynak hashlerini taşır. Başarısız çözüm hedefin
erişilemez olduğunu kanıtlamaz; base-axis sonlu grid kapsamı vardır.

Sayısal motorlar robot-genel refinement/fallback karşılaştırması sağlar.
Eski neural modeller tek başına robot komutu üretmek için kabul edilmedi.
Yeni neural model H2-N1/N2'de oluşturulur; amaç ölçülmüş toplam-bütçe faydasıdır.
Yeni ağın geometrik öğretmenden tüm dal etiketleri alması için mevcut tek-aday
API'sinden farklı, doğrulanmış dal erişimi gerekir; bu hazır varsayılmaz.

Profil A 2 mm/1°, B 1 mm/0,5° değişmez. Final başarısı sonlu iş yükünün ampirik
sonucudur; fiziksel doğruluk, collision, tam uzay veya deadline garantisi değildir.
Geometrik final P95 0,5029 ms, max 103,1543 ms; 60.000/60.000 kinematik başarı,
59.999/60.000 10/50 ms içinde. Eski T-C06 fresh locked runtime; C1-06S clean replay
existing locked runtime kullanmıştır. Bu devir yeni ortam/eğitim/benchmark iddiası taşımaz.

## Sonraki uygulama — H2-01

1. [Görev](../tasks/H2-01.md), iki hashli devir ve [ADR-026](../adr/ADR-026-geometric-core-recovery.md)/[ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md) okunur.
2. `experiments/H2-01/` yeni kanıt alanı açılır; gereksinim REQ-H01/REQ-HG01,
   test T-H01/T-HG01, profil ve hata/kabul protokolü uygulamadan önce kaydedilir.
3. Ortak SolverRequest/Result; robot/model/motor kimliği, q_current/pose/frame,
   toplam bütçe, kontrol kapsamı, collision gereksinimi, invalid/UNRESOLVED/
   LATE_VALID/cancel durumları uygulanır. Collision istenip NOT_CHECKED tam kabul değildir.
4. Kimlik/limit/pose/hata regresyonları koşulur; yanlış kabul 0 gerekir.
   RUN_REPORT, gerçek komutlar/JUnit/hashli kanıt ve acceptance kararı tamamlanır.
5. STATUS/TRACE birlikte güncellenir; H2-01 kabulünden sonra H2-02 açılır.

Sıra H2-01 → H2-02 → H2-N1 → H2-N2 → H2-03 → H2-04 → H2-05 → H2-06.
H1 hedefi P95 ≥%20 iyileşme, başarı kaybı ≤1 yüzde puan ve kayıtlı CI koşullarıdır;
10/50 ms toplam maliyete hazırlık+forward+FK dahil edilir. G2/G-N ayrı karardır.
Yeni seçim/final için bağımsız veri ve erişim protokolü dondurulur. Görülmüş C1,
C1-06S final verisi yeni model seçimi veya yeni final yerine kullanılamaz.
39 ileri görev ve ek kaynakların uyarlaması [makine planı](../roadmaps/ROADMAP_TASKS_r3.json)
ve [kaynak eşlemesinde](../roadmaps/SOURCE_MAPPING_r3.md) bulunur.

## Ortamı hazırlama ve denetleme

Windows 11 x64, Pixi 0.81.0; `pixi.lock` ve `pyproject.toml` korunur.
Foundations kurulumunda [SETUP](../SETUP.md), neural overlay kurulumunda
[tarihsel Core rehberi](archive/pre-hybrid-20261011/docs/records/CORE_RESUME.md)
ve `scripts/reproduce_c107.py`/C1-07 gerçek komut logları esas alınır.
Python 3.12.14, NumPy 2.5.3, Pinocchio 4.1.0; Torch 2.10.0+cu128 overlay.
Neural CUDA decoder testleri yalnız CPU ortamda tamamen geçti varsayılmaz.

```powershell
pixi install --locked
pixi lock --check
pixi run --locked env-check
pixi run --locked python scripts/validate_roadmap_r3.py
pixi run --locked python -m pytest tests/c1_06s -q
```

Geometrik clean replay için ayrı temiz checkout ve arşivlenmiş artifact-root gerekir:

```powershell
$env:PYTHONPATH = (Resolve-Path './src').Path
pixi run --locked python scripts/replay_c106s.py --artifacts 'C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main' --output 'temp/replay-new.json'
```

Bu komut temiz checkout kökünde çalışır; `--artifacts` canlı/devir alınmış kaynak
köküdür, checkout'tan farklı olmalıdır. `temp` klasörünü önce oluştur; yeni çıktı
adı seç. Eski kanıtların üzerine yazma. Protokol kaynak hashleri değişirse replay
hata vermelidir. Ortam testinin geçmesi tüm platformlarda birebir eşlik anlamına gelmez.

## Checkpoint, arşiv ve başka kişiye teslim

Yerel teslim dizini:
`C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-PreHybrid-Checkpoint-20261011`.
Kullanıcı tarafından seçilen ikinci kopya:
`C:\Users\Arda TEKGÖZ\Documents\NK_Arşiv\NeuroKinematics-PreHybrid-Checkpoint-20261011`.
İkisi aynı C: diskindedir; disk arızasına dayanıklı bağımsız aygıt/uzak yedek
**NOT_CONFIRMED**. Gerçek checkpoint SHA ve doğrulama sonucu
[teslim kaydında](../../experiments/PRE-HYBRID-20261011/) bulunur; aşağıdaki tarif
tek başına gerçekleşmiş test iddiası değildir.

`repository.bundle`: commit edilmiş kod/plan/küçük kanıt ve Git geçmişi.
`research-artifacts.zip` + `inventory.json`: `data/generated` (yeni C1-06S dahil)
ve C1-01 büyük JSONL/stderr. Model ağırlıkları normal Git'te değildir.
`user-work/`: önceden mevcut kullanıcı değişiklikleri/ilgisiz belgeler ayrı korunur;
otomatik patch uygulama. `.venv`, `.pixi`, cache/secrets yeniden kurulacak yereldir.
Sealed/final içerik arşiv işlemlerinde yalnız opak bayt olarak saklanır.

```powershell
$nkArchive = 'C:\Users\Arda TEKGÖZ\Documents\NK_Arşiv\NeuroKinematics-PreHybrid-Checkpoint-20261011'
git clone --branch codex/c1-06r "$nkArchive\repository.bundle" NeuroKinematics-Restore
Set-Location NeuroKinematics-Restore
python scripts/archive_core.py verify --archive $nkArchive
python scripts/archive_core.py restore --archive $nkArchive --destination .
```

Restore yalnız boş artifact hedeflerine yazar; mevcut dosyayı ezmez. Önce bundle
doğrulaması, sonra archive/member SHA, sonra boş hedef restore yapılır. Her ikisini
birlikte teslim et; yalnız Git veya yalnız belge model/veri tekrar üretimi için yeterli
değildir. Son doğrulama `scripts/check_pre_hybrid.py` ile yapılır.

## Sınırlar ve kalan iş

Bu checkpoint H2 başlamadan geri dönebileceğimiz durumu korur. Bütün nihai neural
metriklerin sağlandığını, sistemin fiziksel kullanımda stabil olduğunu veya herhangi
bir modelin belgeleri okumadan doğru davranacağını garanti etmez. Başlangıç rehberi
ve kayıtlı görevleri izleyen kişi hangi kabulün tamamlandığını ve sonraki işi görebilir.
Yeni neural/H1/G2/G-N, collision, ONNX ve generic/physical hedefler kendi görevlerinde
ölçülecektir. Bağımsız aygıt yedeği için ayrıca farklı cihaz/uzak hedef gereklidir.
