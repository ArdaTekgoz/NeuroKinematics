# F0-05 sayısal baseline ve ölçüm altyapısı çalışma kaydı

Kimlik: RUN-20260924-F005
Durum: **PASS / TAMAMLANDI**
Görev ve gereksinim: F0-05 · REQ-F05 · T-F08 ve A1/A2 kabulü
Tarih ve sorumlu: 23–24 Eylül 2026 · Codex; proje sahibi Arda Tekgöz
Yazılım hedefi: v0.1.0 · Belge revizyonu: r2

Bu kayıt `docs/templates/RUN_REPORT.md` yapısını izler. Aşama 1 incelemesi
`STAGE1_REVIEW.md`, açık kullanıcı onayı `stage2-approval.json` içindedir.

## Soru ve değişiklik

Sabit damping'li DLS, doğrulanmış F0-02/F0-03 kinematik temeli üzerinde
bağımsız FK ile doğru değerlendiriliyor mu; 12.000 bağımsız hedefe ait 10/50 ms
ve beşer geçişli ölçüm tekrarlanabilir, eksiksiz ve bozulmaya dirençli mi?
`solvers/dls.py`, benchmark sözleşmesi/şeması, query üretici, ölçüm CLI'si,
bağımsız JSONL doğrulayıcısı, özetleyici, analitik dış sınır kanıtı ve fail-fast
kabul runner'ı eklendi. Girdi robotu, kaynak raporlar ve F0-04 korunmuştur.
Yeni bağımlılık veya kinematik mimari değişikliği yoktur; ADR gerekmedi.
Önceden belirlenmiş başarı oranı eşiği yoktur. Kabul matematik, kolay hedef,
yanlış başarı reddi, kapsam, tekrarlanabilirlik ve kanıt bütünlüğüne dayanır.

## Tekrar üretim

Başlangıç `main == origin/main == 84adc24a03f619d7079e4e3900b8f51ab2591ef5`.
Uygulama ve makine-okunur kanıt commit'i `3e55954`.
Kullanıcının Aşama 2 açık onayı kaydedildi; Aşama 1'in 125 testi yeniden geçti.
Dört onaylı JSON hash'i `STAGE1_REVIEW.md` ve `stage2-approval.json` içindedir.
URDF `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96`,
F0-04 içerik `5cb4e64580ecaf99afd11b3c8b98e06ed00c712e83bf2d9ee4d8c3acd58173fe`,
`pixi.lock` `56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a`.
Diğer değişmez girdiler `preflight.json` içindedir. PCG64 ve seedler frozen
`config.json` içindedir. Model ağırlığı/checkpoint: **UYGULANMAZ**.

Windows 11 x64, Python 3.12.14, NumPy 2.5.3; 16 mantıksal CPU, seri CPU koşusu.
GPU kullanılmadı; RAM ve gerçek thread sayısı `NOT_MEASURED`; Linux `NOT_RUN`.
Ölçüm saati `perf_counter_ns`. Başlangıç/bitiş UTC zamanları `commands.json`,
`solver-summary.json` ve `acceptance.json` içindedir; etkin insan emek süresi
`NOT_MEASURED`. Ana komut `pixi run --locked accept-f05`; tam alt komut ve
ek hedefli test kaydı `COMMANDS.md` ve `commands.json` içindedir.

Tam query ve sonuç JSONL Git dışında `data/generated/F0-05/acceptance/run-a/`
altındadır. Query SHA-256
`120b41f07109aeaca10e10fbb04783167cdc4ba282e7976941468c7dccfa4976`;
iki bağımsız üretim aynı hash'i verdi. Sonuç JSONL SHA-256
`b18161e53ca3ced0266d825afef7d9535036b9ed3d5617606ea6290f1926a724`.
Dosya yolları, kayıt sayıları ve tekrar üretim komutları `query-manifest.json`
ve `result-manifest.json` içindedir. Timing satırlarının bayt düzeyinde
deterministik olması beklenmez.

## Test ve ham kanıt

| Gereksinim | Değişiklik | Test | Son kanıt |
|---|---|---|---|
| DLS ve A/B bağımsız FK | `solvers/`, `benchmark/validation.py` | F0-05 unit 127/127; T-F08 16/16 | `f05-unit-junit.xml`, `tf08-junit.xml` |
| Query bağımsızlığı ve determinism | `benchmark/queries.py` | iki üretim ve tam F0-04 karşılaştırması | `query-hashes.json`, `query-manifest.json` |
| Ölçüm, deadline, JSONL | `benchmark/runner.py`, CLI | 120.000 satır bağımsız tam doğrulama | `solver-summary.json`, `result-verification.json` |
| Bozuk/erişilemez/çözümsüz girdiler | proof, validator | T-F08, analitik sertifika doğrulaması | `failure-cases.json`, T-F08 JUnit |
| Kabul kapısının hata reddi | production mutasyonları | 32/32 algılandı | `mutation-results.json`, `mutation-junit.xml` |
| Önceki Foundations temeli | Pixi ve runner | F0-00/01/02/03/04: 6/16/102/159/39 PASS | ilgili `f00`–`f04-junit.xml` |
| Tamlık ve hash | manifestler, `SHA256SUMS` | query/result SHA ve kanıt verify-only PASS | `evidence-verification.json` |

12.000 benzersiz hedef: main 10.000, boundary 1.000, singularity-near 1.000.
Local/wide sırasıyla 5.000/5.000, 500/500, 500/500. F0-04 ile tam q tekrarı,
grup kesişimi, subsetler arası tekrar ve ID çakışması sıfır. `q_target` solver
API'sine verilmez; her query aynı 10/50 ms listesinde beş kez ölçülür. 40
warm-up çağrısı, 0,354 s query yükleme ve 3,003 s serialization/IO ayrı tutuldu.
Joint-limit ihlali sıfır; collision her satırda `NOT_CHECKED`.

## Sonuç ve yorum

Her deadline için 60.000 deneme vardır; beş tekrar yeni bağımsız hedef sayılmaz.
Geometri ve toplam süreye bağlı deadline başarı oranları:

| Profil | 10 ms geometri | 10 ms deadline | 50 ms geometri | 50 ms deadline |
|---|---:|---:|---:|---:|
| A (2 mm, 1°) | 51,027% | 50,462% | 69,272% | 68,768% |
| B (1 mm, 0,5°) | 50,867% | 50,462% | 68,677% | 68,627% |

50 ms B için main/local %100, main/wide %38,088, boundary/local %100,
boundary/wide %33,560, singularity/local %100 ve singularity/wide %32,600
deadline başarısı. Geniş başlangıçtaki düşük oran sabit-damping yerel DLS
baseline sonucudur; başarısızlığa göre kabul eşiği değiştirilmedi.

| Deadline | Tüm denemeler latency P50/P95/P99 (ms) | B deadline başarıları latency P50/P95/P99 (ms) | İterasyon P50/P95/P99 |
|---|---|---|---|
| 10 ms | 7,843 / 11,719 / 12,249 | 3,305 / 6,286 / 7,739 | 3 / 11 / 13 |
| 50 ms | 5,857 / 50,656 / 51,044 | 2,014 / 21,427 / 34,087 | 3,5 / 64 / 67 |

Başarısız denemelerin süreleri tüm denemeler dağılımına dahildir. 10 ms'de
konum hata P50/P95/P99 0,000968/0,966825/1,225599 m; yönelim
0,122/133,086/159,556°. 50 ms'de sırasıyla
0,000671/0,585618/0,906770 m ve 0,044/99,218/148,303°.
50 ms B main/local konum P50/P95/P99 0,000461/0,000916/0,000981 m,
main/wide 0,014834/0,762932/0,981922 m. Hard subset boundary/wide
0,043406/0,829100/1,046176 m, singularity/wide
0,005118/0,795635/1,048032 m. Tam dağılımlar, max değerleri ve
orientasyon dahil `subgroup-summary.json` ve `deadline-summary.json` içindedir.

120.000 sonuçta `SUCCESS` 71.521, `TIMEOUT` 40.486, `STALLED` 7.993;
`MAX_ITERATIONS`, `NUMERICAL_FAILURE`, `INVALID_INPUT` üretim querylerinde
sıfır. T-F08 bozuk girdi için `INVALID_INPUT`, analitik dış hedef için bağımsız
sertifikayla `PROVEN_UNREACHABLE`, kanıtsız yakınsamama için `UNRESOLVED`
kanıtlar. Dış hedef `100 m` ve zincir translation norm üst sınırı `1,415 m`.
Solver başarısızlığı erişilemezlik kanıtı değildir.

Tam kabul koşusu PASS ardından proof raporlamasında son sınıflandırma
assert'i eklendi; T-F08 yeniden 16/16 geçti, aynı sonuç JSONL'den
`failure-cases.json` yenilendi ve son checksum doğrulaması yapıldı.
Kabul performansı yeniden ölçülmedi. Yazılım yalnız model-içi kinematik
geçerliliği sınar; çarpışmasızlık veya fiziksel robot güvenliği ölçülmedi.

## Sonraki adım

F0-05 girdileri ve kanıt hashleri F0-06'nın temiz kurulum/faz devir
incelemesine hazırdır. F0-06 başlatılmadı ve G0 kapanmadı. Geniş başlangıç
başarıları, Linux koşusu, RAM/thread ölçümleri ve fiziksel güvenlik bu görevin
kanıtından çıkarılamaz. İlgisiz Word lock dosyası kapsam dışında bırakıldı.
