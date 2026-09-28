# C1-02 Aşama 1 çalışma kaydı

Kimlik: RUN-20260928-C102-STAGE1<br>
Durum: **IN_PROGRESS / STAGE_1_COMPLETE**<br>
Görev ve gereksinim: C1-02 / REQ-C03; T-C07 **NOT_RUN**<br>
Tarih ve sorumlu: 28 Eylül 2026 · Codex; proje sahibi Arda Tekgöz

## Soru ve değişiklik

F0-04 değişmez kök verisinden, G0 robot/TCP sözleşmesini ve F0-05/C1-01 bağımsız benchmark sınırını koruyarak `local`/`wide` çiftlerin yeniden üretilebilir sözleşmesi dondurulabilir mi? Yeni [config](config.json), [schema taslağı](schema.json), [tasarım ve test matrisi](STAGE1_REVIEW.md), [hash manifesti](input-hashes.json), [komutlar](COMMANDS.md) ve salt okunur ön kontrol eklendi. Üretim veya kapsamlı öğretmen etiketlemesi yapılmadı. Önceden kabul edilmiş F0-04/05/C1-01 dosyaları değiştirilmedi. Mimari/kapsam sapması yok; ADR gerekmiyor.

## Tekrar üretim

Başlangıç `main` HEAD ve `origin/main`: `a645153c772a2203994bc3e58a36b50ccb458eae`. Başlangıç çalışma ağacında ilgisiz iki izlenmeyen Word dosyası vardı; commit dışı tutuldu. Yazılım hedefi v1.0.0, bu belge revizyonu r1. Ortam bu aşamada host Windows x64, `python` ile standart kütüphane ön kontrolüdür; tam Pixi/NumPy/Pinocchio runtime **NOT_RUN**. Kanonik veri üretimi için `pixi.lock` SHA `56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a`; C1-01 Ubuntu/ROS runtime lock ayrı ve yeni veri işi için zorunlu sayılmadı. CPU/GPU/RAM/etkin insan emek süresi **NOT_MEASURED**. Eğitim seed'i **NOT_ASSIGNED**; veri seed'i configte `2026092802`.

Gerçek sıra ve çıkış kodları:

| Komut | Sonuç |
|---|---|
| `python scripts/check_c102_stage1.py --write-manifest` ilk deneme | exit 1; şema/model girdi alan sırasını gereksiz eşitleyen kontrol hatası; düzeltildi |
| `python scripts/check_c102_stage1.py --self-test` ilk deneme | exit 1; aynı hata; düzeltildi |
| `python scripts/check_c102_stage1.py --write-manifest` | exit 0; 43 dosya hash PASS, 12 kaynak shard |
| `python scripts/check_c102_stage1.py --self-test` | exit 0; 4/4 bozuk sözleşme mutasyonu yakalandı |
| `python scripts/check_c102_stage1.py --check` | exit 0; frozen hash manifest eş; [ham çıktı](stage1-check.json) |

Son self-test ham çıktısı [stage1-self-test.json](stage1-self-test.json); ilk başarısız konsol çıktısı bu oturum araç kaydındadır. Yeniden çalıştırma tarifi [COMMANDS.md](COMMANDS.md). Başlangıç ve bitiş saatleri/duvar süresi ayrıca ölçülmedi; etkin emek iddiası yok. Üretim dosyası/model/checkpoint SHA **YOK**.

## Test ve ham kanıt

| Gereksinim | Değişiklik | Gerçek kontrol | Kanıt ve durum |
|---|---|---|---|
| G0 değişmez robot/veri kimliği | hash manifest | 15 handoff + 17 C1-01 frozen + 12 shard; birleşik 43 dosya | [input-hashes.json](input-hashes.json), `--check` PASS; tekrar FK NOT_RUN |
| Kök önce split, local/wide 50/50, etiket/girdi ayrımı | config/schema taslağı | statik tutarlılık ve 4 in-memory mutation | `--self-test` PASS; gerçek çift **NOT_RUN** |
| Teacher aday, bütçe, geçerlilik, bias/null | config ve stage review | statik budget hesabı | 90 wide/360 çağrı/72.000 iterasyon plan; pilot **NOT_RUN** |
| Kaynak aile, benchmark izolasyonu | audit matrisi | manifest/query hash | F0-05 query list SHA eş; yeni çift-query overlap **NOT_RUN** |
| T-C07 kabul/regresyon | test matrisi | yok | **NOT_RUN** |

G0/F0-04 geçmiş PASS kayıtları bu oturumun yeni test sayısı değildir. C1-01 600.000 raw yerel dosyası var, bu görevde satır satır okunmadı veya öğretmen diye kullanılmadı. Yeni üretilen local/wide/split, geçerli, etiketsiz, elenen satırların tümü **0 üretilmiş / planlanmış sayılar ölçülmemiş**. Hash tutarsızlığı gözlenmedi. Büyük veri arşiv politikası config ve [review](STAGE1_REVIEW.md) içinde.

## Sonuç ve yorum

Sözleşme ve kaynak byte kimlikleri donduruldu. Bu, C1-02 veri kümesi veya T-C07 geçişi değildir. Local yakın hedefleri sentetiktir; trajectory/ivme/jerk anlamı yoktur. Wide öğretmen başarısı, maliyeti ve seçim dalı dağılımı bilinmiyor. Çarpışma ve fiziksel robot güvenliği **NOT_CHECKED**. Bu raporu içeren teslim commit'i `git log -1 --format=%H -- experiments/C1-02/RUN_REPORT.md` ile okunur; push ve remote SHA eşitliği committen sonra ayrıca doğrulanır. Kendi commit SHA'sını rapora yazarak döngü yaratılmaz.

## Sonraki adım

Yalnız kullanıcının **açık Aşama 2 onayı** sonrası frozen hashleri yeniden doğrula; loader/split/unit → 90 wide pilot → gerekirse tam üretim → T-C07 kapı sırasını uygula. C1-02 PASS/COMPLETE kararı verilmedi. C1-04 için ayrıca C1-03 kabulü gerekir; bu görevde başlanmaz.
