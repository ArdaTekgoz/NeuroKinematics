# C1-06 erişim ve tekrar üretim

8 Ekim 2026 · Yerel kök start.json içinde. Uzak büyük veri arşivi NOT_CONFIRMED.

input-hashes.json: 322 gerçek dosyanın yolu/baytı/SHA/erişimi; tümü PASS.
21 best checkpoint: C1-05'ten 18, C1-04'ten üç conditioned kontrol. Bunlar
LOCAL_ONLY; Git checkpointleri veya veri shardlarını taşımaz.
34 C1-02 shardının tamamı byte hash ile doğrulandı; test içerikleri açılmadı.

C1-01 udp-v2/full altındaki beş ham dosya toplam 1.112.301.428 bayt,
600.000 newline/satır; her biri 120.000. Dosyaların gerçek SHA'ları
input-hashes.json/baseline içinde, geçmiş full ve verify gate'leriyle eş.
Stage 1 yalnız sayım/hash yaptığı için yeni satır semantik doğrulaması iddia edilmez.
12.000 sorguluk query-list de gerçek byte hash/count ile geçmiş manifeste bağlandı.

Başka makineye taşırken listedeki LOCAL_ONLY dosyaları göreli depo yollarıyla
ayrıca kopyala, ardından check_c106_stage1.py --check çalıştır. Manifestte
accessible=true bu makinede doğrulanan erişimdir; uzakta saklandığı anlamına gelmez.
Checkpointlerin config/normalizasyon/robot ve kaynak kodu birlikte taşınmalıdır.

Eksik dosya halinde:

- Baseline: önce özgün udp-v2 ham dosyalarını doğrulanmış kopyadan geri getir.
  Kopya yoksa C1-01 COMMANDS ve run_c101_session.ps1 ile yeni isimli oturumda
  prepare → smoke → pilot → full → verify gerekir. Geçmiş full duvar süresi
  18,4737 saat; yeniden çalışma maliyeti ölçülmemiştir. Yeni süre/raw SHA eski
  kanıt yerine geçirilmez, yeni kaynak sürümü ve protokol revizyonu gerekir.
- C1-02 shardları: experiments/C1-02/COMMANDS.md ve hashli üretim config'i.
  Geçmiş iki üretim yaklaşık 2.336/2.375 saniye; yeni üretim NOT_RUN.
- Checkpoint: öncelik SHA'sı aynı özgün dosyanın kurtarılmasıdır. Yeniden eğitim
  otomatik olarak aynı ağırlık veya kabul edilen final aday değildir; yeni deney
  kaydı gerekir. C1-05 geçmiş toplam duvar süresi yaklaşık 1.496 saniyedir;
  temiz ortamda eğitim tekrarı NOT_RUN'dır.
- Query list: F0-05 query-manifest içindeki exact generate-f05 komutu ve aynı
  kaynak F0-04 girdileri; hash tam eşleşmeden benchmark bağlanamaz. Aşama 1'de
  yeniden üretim yapılmaz; byte kaynak erişilebilir ve doğrulanmıştır.

Yeni C1-06 raw için data/generated/C1-06/<benzersiz-run>/ kullanılır;
252.000 tekil model/query sonucu × 5 süre geçişi = 1.260.000 satır planlıdır.
Gerçek boyut ve süre henüz NOT_MEASURED. Küçük rapor/özet/CI/hata/manifest/log
experiments/C1-06/stage2/<run>/ altında Git'e alınır. Her raw dosya için
bytes/rows/SHA/access/storage/remote_archive yazılmadan T-C05 kapatılamaz.

Bu çalışma sırasında birinci input audit yanlış witness dosya adı nedeniyle
FAIL verdi. input-audit-attempt-001.json ve commands/input-audit korunmuştur.
Doğru mevcut fixed-validation-witness.json adıyla ikinci deneme PASS oldu;
hiçbir girdi veya ağırlık değiştirilmedi. Test/benchmark satırı okunmadı.
