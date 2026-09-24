# F0-06 kapanış ve faz devri çalışma kaydı

Kimlik: RUN-20260924-F006
Durum: PASS / TAMAMLANDI
Görev ve gereksinim: F0-06 · REQ-F06 · T-F09
Tarih ve sorumlu: 24 Eylül 2026 · Codex; proje sahibi Arda Tekgöz
Yazılım hedefi: v0.1.0 · Belge revizyonu: r1

## Soru ve değişiklik

F0-00–F0-05 kanıt zinciri korunarak kilitli temiz Windows ortamında gerçek küçük
veri/query/DLS benchmark zinciri iki kez üretildi. Üretim configleri, solver,
kinematik eşikler ve eski kanıt dosyaları değiştirilmedi. Ayrı smoke girdilerini
kabul eden public API parametreleri, fail-closed kapanış denetimleri ve kayıt
scriptleri eklendi; gerekçe [ADR-006](../../docs/adr/ADR-006-foundations-temiz-yeniden-uretim.md).

## Tekrar üretim

- Başlangıç main = origin/main = `e7d211f42496f803688e2a510daca97e102092dc`.
- Beklenen F0-05 uygulama zinciri `3e559548e417571d83a7b44315485a1d43559120` → `e7d211f42496f803688e2a510daca97e102092dc` doğrulandı.
- F0-06 uygulama commit'i `8e53698a414d34e96039e64a48c153e7decfb7b1`.
- Kapanış commit'i bu raporu ve G0 kararını ekleyen ikinci committir; kendi hash'ini içine gömmez. `git log -1 -- experiments/F0-06/G0_DECISION.md` ile çözülür; push sonrası SHA kullanıcı yanıtında verilir.
- Kullanıcıya ait Word lock dosyası korunmuş, commit dışında bırakılmıştır.
- Kanonik temiz worktree: `C:/Users/Public/NeuroKinematics-F006-verification`; başlangıç commit'i üstteki F0-05 kapanışıdır. F0-06 farkı `overlay.json` ile açıkça hashlenmiştir.
- İlk `.pixi`, `.pytest_cache`, `data/generated` ve F0-06 çıktı yollarının yokluğu kaydedildi. Önceki ortam/veri/cache kopyalanmadı. Global Pixi paket indirme önbelleği kullanıldı; cache-free iddiası yoktur.
- `pixi install --locked`, `pixi lock --check`: exit 0. Lock SHA-256 `56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a`; değişmedi.
- Windows 11 10.0.26200 AMD64; Pixi 0.81.0; Python 3.12.14; NumPy 2.5.3; Pinocchio 4.1.0. pytest/hatchling/xacro/PyYAML sürümleri `canonical/logs/runtime.stdout.log` içinde. CPU/seri çalışma `measurement-summary.json`; GPU kullanılmadı, RAM/gerçek runtime thread sayısı NOT_MEASURED.
- Config ve seedler koşudan önce hashlenmiş `config.json`, `data-config.json`, `benchmark-config.json` içinde. Veri/query sayıları her biri 256 main + 64 boundary + 64 singularity. Deadline 10/50 ms, birer geçiş, deadline başına dört warm-up.
- Robot/TCP/config/schema hashleri `handoff-inputs.json`. Model/checkpoint: UYGULANMAZ; öğrenme başlatılmadı.
- UTC kabul aralığı: 2026-09-23T22:11:51.075405+00:00 → 2026-09-23T22:13:17.172018+00:00. Etkin insan emek süresi NOT_MEASURED.
- Tam komut, cwd, exit code, stdout/stderr SHA ve JUnit sayıları [COMMANDS.md](COMMANDS.md), `commands.json` içinde.

## Test ve ham kanıt

| Gereksinim / kontrol | Gerçek sonuç | Kanıt |
|---|---|---|
| Başlangıç kimliği | main/origin/expected aynı | preflight.json, canonical/logs/head.stdout.log |
| F0-00–F0-05 bütünlüğü | 212 SHA256SUMS kaydı, 18 Stage 1 kaydı, 29 robot manifest dosyası | history-audit.json, stage1-audit.json |
| Üretim büyük dosyaları | 12 F0-04 shard + F0-05 query/result, 14/14 yerel hash PASS; Git dışında | history-audit.json external_evidence |
| Tarihsel rapor/JUnit sayıları | 13 ana karşılaştırma PASS; eski geliştirme hataları ayrı tutuldu | history-audit.json |
| T-F00–T-F08 resmi regresyonlar | F0-00/01/02/03/04: 6/16/102/159/39; F0-05 unit 127; T-F08 16; mutasyon 32; toplam 497 PASS, 0 hata/skip | canonical/regression/junit/ |
| T-F09 gerçek iki üretim | Her koşuda 384 veri, 384 query, 768 gerçek DLS sonuç satırı | canonical/reproduction/a ve b |
| Determinizm | Dataset/query hashleri ve 6 shard file/content hashleri aynı | reproduction-summary.json |
| Split ve duplicate | Tüm alt kümeler birleştirilerek grup kesişimi ve çapraz split exact q tekrarı sıfır | dataset-audit.json, split/duplicate-audit.json |
| Train-only normalizasyon | Shardlardan yeniden hesaplandı; validation/test değerlerinin değişimi sonucu etkilemedi | dataset-audit.json, normalization.json |
| FK, schema ve limitler | İki koşu PASS; boundary ve singularity kuralları korundu | dataset-audit.json, result-verification.json |
| Benchmark sözleşmesi | Kimlik/deadline/config/seed alanları eş; her satır schema + bağımsız FK/limit doğrulandı | reproduction-summary.json, result-verification.json |
| F0-06 kapanış testleri | 26/26 PASS: 1 pozitif + 25 negatif; 24 açık mutasyon property kaydı + frozen-rule negatif testi | mutation-results.json, canonical/regression/junit/f06.xml |
| İndeks yapısı | Tam alan/test/görev yapısı PASS; eksik T-F09 mutasyonu reddedildi | evidence-verification.json |
| Final checksum | SHA256SUMS deterministik sıralı; kendisini dışlar; dosya adedi evidence-verification.json içinde | SHA256SUMS |

Dataset SHA-256: `0f5e08c735ca1e4432700d93f6d649a5718e612d548ca387893953ee9b66da69`.
Query SHA-256: `047938c1e16f156a32cf8ff25a88e7bf3105e1504fa61669892080f1922a8afa`.
Kanonik sonuç SHA-256 A: `faf12e76eae17c20044a9d6d53ad4397047f6a281274ef12bbb72bb54de10cff`;
B: `8a52b57801c8e8629ae35793c54ae53eed9e47a88faf258557ab1a5044836f0d`.
Ham benchmark hashlerinin aynı olması beklenmez: Windows scheduler/host süreleri,
iterasyon sayısını ve deadline'a bağlı sonucu etkileyebilir. Deterministik alanlar
ayrı projeksiyonla eşleştirildi. Manifest üretim yolu farklıdır; her sonuç kendi
ham manifest hash'ine doğrulanır, iki manifest byte-identical diye sunulmaz.

## Sonuç ve yorum

T-F09 PASS; G0 PASS / ACCEPTED; Foundations COMPLETE; Core READY / NOT_STARTED.
Kritik açık hata yoktur. G0 kararı [G0_DECISION.md](G0_DECISION.md), devir
[CORE_HANDOFF.md](CORE_HANDOFF.md) içindedir. Başarı oranına göre eşik seçilmedi.

Belge/kanıt ayrımları: İstenen `docs/MASTER_ROADMAP.md` başlangıçta yoktu;
bağlayıcı ana roadmap `docs/roadmaps/MASTER_ROADMAP.md` idi. Kök docs yolu için
ona işaret eden giriş eklendi. REQ-F00 depoda tanımlı değildir; F0-00 ve F0-01,
REQ-F01'e bağlıdır. Eski teknik rapordaki Linux önerisi ADR-004 ile Windows
kanonik kararına dönüştürülmüştür; Linux doğrulanmış sayılmaz.

Geçmiş shared pixi/SETUP/gitattributes hash farkları kendi commitlerindeki ham
bloblarla doğrulandı; immutable girdiler için istisna yoktur. F0-06 public API
kaynak farkları baseline blob hash'i ve overlay hash'iyle ayrı kayıtlıdır.
F0-00 iki JSON dosyasının tarihsel Git blob satır sonları checkout baytlarından
farklıdır; checksumla eşleşen mevcut ham dosyalar doğrulandı, blob eşitliği
iddia edilmedi. İlk audit denemesinde F0-06 kaynak farkları eski hashlerle
karşılaştırıldığı için üç fark görüldü; `development-history-audit.json` saklandı,
baseline-vs-overlay ayrımı açıklanarak doğru sürümler ayrı doğrulandı.

İlk çalışma alanının logları korunur; kanonik kabul ikinci, tamamen yeni alanda
ilk kurulumdan itibaren zaman damgalıdır. Ham log/JUnit CRLF/terminal boşlukları
normalize edilmedi; yalnız bu capture dosyalarının Git whitespace davranışı
`.gitattributes` ile tanımlandı. Sayısal kabul eşikleri değişmedi.

- Linux NOT_RUN; kilitte linux-64 bulunması yürütme kanıtı değildir.
- Fiziksel robot doğruluğu, kalibrasyon, collision checking ve gerçek robot güvenliği doğrulanmadı.
- F0-05/F0-06 süre ölçümleri bu Windows hostuna özgüdür; gerçek zaman garantisi yoktur.
- DLS geniş başlangıç performansı sınırlıdır; ölçülen haliyle baseline kalır. Başarı oranı eşiği türetilmedi.
- Coverage ampiriktir; matematiksel tam kapsama/erişilebilirlik garantisi değildir.
- Harici solver baseline'ları Core kapsamındadır. Eğitilmiş neural model yoktur.
- Üretici PDF raw bayt/hash eksikliği önceki görevlerden devralınmıştır.
- Foundations kapanışı üretim veya fiziksel robot güvenlik onayı değildir.

## Sonraki adım

C1-01, C1-02 ve C1-03 G0 sonrasında uygun; üçü de NOT_STARTED.
Roadmap sırası korunur, ek öncelik uydurulmadı. Core önce hashli devir girdilerini
kontrol etmeli; kendi testleri geçmeden Torch FK/harici solver/model doğruluğu
iddia etmemelidir. Bu görev içinde Core işi, Git tag veya release oluşturulmadı.
