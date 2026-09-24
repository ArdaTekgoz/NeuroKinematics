# Deney veya uygulama kaydı

Kimlik: RUN-20260924-C101-S1
Durum: IN_PROGRESS / STAGE_1_COMPLETE; T-C00 NOT_RUN
Görev ve gereksinim: C1-01 · REQ-C01
Tarih ve sorumlu: 24 Eylül 2026 · Codex; proje sahibi Arda Tekgöz

## Soru ve değişiklik

Soru: F0-06/G0 girdileri korunurken dört harici IK varyantı mevcut DLS ile karşılaştırılabilir ortak sözleşmede nasıl entegre edilecek? Bu oturumun tek ana farkı kaynak/platform/adapter/benchmark sözleşmesinin sonuç görmeden dondurulmasıdır. Beklenen Stage 1 kabulü hash, config, solver ayrımı, negatif kontroller ve F0 regresyonlarıdır. Solver matematiksel başarı eşiği Stage 1'de ölçülmez.

## Tekrar üretim

Başlangıç branch/HEAD: `main` / `e1971bf8b70154d6f4d882546c20de0eaab2d83d`. Başlangıç `origin/main` aynı SHA. İki ilgisiz, izlenmeyen Word dosyası vardı ve korunuyor. Kapanış commit'i bu kayıtla aynı commit olduğundan içine SHA yazılarak döngü oluşturulmadı; `git log -1 --format=%H -- experiments/C1-01/RUN_REPORT.md` ile bulunur. Yazılım halen `0.1.0`; hedef `v1.0.0`, belge revizyonları ayrı güncellendi.

Gerçek yürütme Microsoft Windows 11 Pro 10.0.26200 x64, AMD Ryzen 7 250 (8 çekirdek/16 mantıksal), 23.31 GiB görünür RAM; Python 3.11.9 sistem, Pixi 0.81.0 ve mevcut kilitli `.pixi` ortamı. GPU kullanımı ve etkin insan emeği `NOT_MEASURED`; yeni ROS/C++ runtime, compiler, CPU seti ve Linux `NOT_AVAILABLE`. Stage 2 hedefi Ubuntu 24.04 LTS x86_64 / ROS 2 Jazzy / MoveIt 2.15.2; henüz kurulmadı. Robot/TCP hashleri ve 15 immutable dosya [frozen-hashes.json](frozen-hashes.json); query-list SHA `120b41f07109aeaca10e10fbb04783167cdc4ba282e7976941468c7dccfa4976`. Config ve beş exact solver pini [baseline-config.json](baseline-config.json); source/license [DEPENDENCIES.md](DEPENDENCIES.md). Seedler F0-05 query configinden devralınır; pick_ik global determinism henüz `NOT_AVAILABLE`. Gerçek komutlar [COMMANDS.md](COMMANDS.md). Başlangıç/bitiş duvar saati ve download beklemesi kaydedilmedi; insan emeği tahmin edilmez.

## Test ve ham kanıt

| Test | Girdi/çıktı | Gerçek sonuç | Durum |
|---|---|---|---|
| G0/handoff/hash kapısı | F0-06 karar, handoff ve 15 immutable dosya | 15/15 SHA; G0 PASS / ACCEPTED | PASS |
| C1-01 Stage 1 config ve mutasyon | `baseline-config.json`, F0-05 config/query; `stage1-verification.json` | 52/52 PASS; 15/15 negatif mutasyon yakalandı | PASS |
| İlk toplu F0 pytest çağrısı | `stage1-f0-regression.log`, `.xml` | 5 collection error; exit 2; test execution başlamadı | FAIL; ham kanıt korundu |
| F0-00–F0-06 ayrı regresyon | 7 ayrı `.log` / `.xml` | 6/16/102/159/39/175/26 = 523 PASS, 0 FAIL | PASS |
| T-C00 solver ve tam benchmark | `NOT_AVAILABLE` | `NOT_RUN`; 0 harici solver çalıştırması; 0 yeni benchmark satırı | NOT_RUN |

Stage 1 ham kanıt dosyalarının SHA-256 değerleri [evidence-hashes.json](evidence-hashes.json) içindedir. Testler matematiksel harici baseline başarısını veya Linux yürütmesini kanıtlamaz. F0-05’in tarihsel 120.000 satırlı Windows sonucu değiştirilmedi.

## Sonuç ve yorum

Sözleşme donduruldu; C1-01 **IN_PROGRESS / STAGE_1_COMPLETE**, **PASS/COMPLETE değil**. Beş solver kimliği kayıtlıdır ama yalnız DLS'in F0-05 tarihsel sonucu vardır. KDL, TRAC-IK ve pick_ik local/global entegrasyonu `NOT_RUN`; yeni başarı/gecikme `NOT_MEASURED`. Aynı Linux hostta DLS tekrar koşmadan Windows/Linux hız kıyası yapılamaz. Collision `NOT_CHECKED`; fiziksel robot/kalibrasyon/güvenlik test edilmedi. pick_ik deprecated, yalnız temel bakımdadır. ROS paket revizyonları ve runtime lock'u Linux ortamı olmadığı için açık Stage 2 kapısıdır. DLS depo genel lisansı `NOT_DECLARED`.

## Sonraki adım

Açık kullanıcı onayı sonrası tek ana iş: kilitli Linux/Jazzy ortamını kurup ortak adapter ve dört harici solver varyantını kolay smoke ile çalıştırmak. Smoke geçmeden tam T-C00 benchmark yapılmaz. 20 saat etkin emek sınırı gerçek çalışma aralıklarıyla izlenir. C1-02/03/06 başlatılmaz.
