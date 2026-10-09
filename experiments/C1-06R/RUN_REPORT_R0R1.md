# Deney veya uygulama kaydı

Kimlik: RUN-20261009-C106R-IMPLEMENTATION
Durum: **R0 PASS / R1 DIAGNOSTIC_COMPLETE_WITH_RECORDED_FAILURES / READY_FOR_USER_TRAINING**
Görev ve gereksinim: C1-06R / REQ-C02–05; C1-07 için hazırlık
Tarih ve sorumlu: 9 Ekim 2026, Codex; proje sahibi Arda Tekgöz
Yazılım hedefi v1.0.0; belge r1. Core/G1 kabulü ve sürüm etiketi verilmedi.

## Soru ve değişiklik

Kullanıcı planın ardından “Tamam işlemlere başlayalım”, daha sonra “İşleme
kaldığın yerden devam et” dedi. Ayrı GPU ortamı ve tanılar tamamlandı; ilk
uzun validation kampanyası çalıştırılabilir paket olarak hazırlandı. Uzun
eğitim kullanıcının işi olarak korunmuştur. [ADR-014](../../docs/adr/ADR-014-c106r-independent-research.md)
ve [görev kaydı](../../docs/tasks/C1-06R.md) yeni kapsamı eski C1-06'dan ayırır.

Yeni kod: `c106r.py` tanılar, `c106r_training.py` dört arm/eşli eğitim ve
hashli atomik resume; CLI, launcher, paket denetimi ve freeze scriptleri.
Mevcut model, veri, G0 veya eski öğrenme/FK kodu değiştirilmedi. Yeni eğitim
seçimi: tam validation paydasında A başarısı → geçersiz sayısı → normalize
poz hatası → eşitlikte erken epoch. Eski H2 ve checkpoint seçimi korunur.

## Tekrar üretim

Başlangıç HEAD/main/yerel origin/main:
`77dae3844d60bbc242855d3874785facf0f1f786`. Remote fetch yapılmadı.
Çalışma ağacında kullanıcıya ait STATUS/TRACEABILITY ve izlenmeyen PDF/Word/
AUDIT kayıtları vardı. STATUS/TRACEABILITY'ye yalnız C1-06R eki yazılır;
önceki byte prefix'i `user-prefix-preservation.json` ile korunur. İlgisiz
dosyalar commit kapsamına alınmaz. Yazılım paket kimliği [training-freeze](training-freeze.json),
Git teslimi `delivery.json` içindedir.

Windows 11, RTX 5060 Laptop (compute capability 12.0), 8151 MiB VRAM,
kullanıcı beyanıyla 24 GB RAM. Python 3.12.14, Torch 2.10.0+cu128;
NumPy ve yardımcı paketler C1-03 ile aynı sürümlerin exact wheel'leri.
`.venv/c106r` ayrı overlay; G0 `pixi.lock` ve `.venv/c103` korunur.
11 paket için sürüm/kurulum yolu/direct_url/hash `runtime.json` içinde.
TF32/AMP kapalı, deterministic algorithms ve CUBLAS workspace açık.

Paket seçimi [resmi PyTorch sürüm listesi](https://pytorch.org/get-started/previous-versions/)
ve [CUDA 12.8 Blackwell desteği](https://pytorch.org/blog/pytorch-2-7/) ile
kontrol edildi; gerçek runtime testi ayrıca koşuldu. CUDA wheel indirme/kurulum
yaklaşık 10 dakika 37 saniye sürdü, exit0. İlk gözlemde ilerleme görünmüyordu;
kullanıcının devam mesajından sonra tamamlandığı görüldü. Alternatif paket
kurulmadı; sadece resmi alternatif URL'ye küçük HTTP range erişim denemesi yapıldı.

Komutlar `commands/*/command.json` içinde argv/cwd/UTC/exit/stdout/stderr SHA
ile kayıtlıdır. `001–003` resolution/venv/install; `004` CPU tanı testleri;
`005–006` preflight/veri; `007–008` pip/GPU; `009` ilk overfit; `010` başarısız
toplu collection; `011-*` ayrı suite regresyonları; `012–013` iki ayrı optimizer
tanısı; `014` eski validation logları; `015` resume testleri; `016` eğitim smoke;
`017` freeze; `018` gerçek PowerShell launcher `-CheckOnly`.

Ölçüm öncesi tanı koşulları `r0r1-config.json`; sonraki tanılar
`overfit-followup-config.json` ve `overfit-scale-config.json` ile ayrı kayıtlı.
Her sonucun config/kod/weight SHA'sı saklandı. Uzun eğitim protokolü
`training-round1.json`: aynı veri/kapasite, 2000 epoch, batch256, 120.000
adım/arm-seed, 4 arm × 3 seed. İki-epoch smoke ana eğitim değildir.

## Test ve ham kanıt

| Gereksinim / tanı | Gerçek sonuç | Kanıt |
|---|---|---|
| G0 ve seçili veri byte/SHA | PASS; robot/TCP/lock sabit; yalnız train/validation shardları açıldı | r0r1/attempt-001/preflight.json |
| CPU/GPU FK | 1086 q × 2 dtype × 2 cihaz, her örnek mevcut toleranslarda PASS | r0r1/attempt-001/runtime.json |
| GPU/CPU gradyan | Her cihazda 32 q × 12 çıktı × 6 türev; bağımsız Pinocchio FD PASS | runtime.json gradients |
| Eğitim kaybı/başlık | Q/p/R türevleri, tanh round-trip/türev ve FK-only parametre güncellemesi PASS | runtime.json physics_probe |
| Veri/etiket | 16.800 train + 3.600 validation, 18.453 mevcut etiketin tamamı B geçerli; 1.947 eksik wide korunur | data-audit.json |
| Provenance/normalizasyon | 20.400 q_current tam tekrar; root FK ve train-only istatistik PASS; train/validation root/group kesişimi0 | data-audit.json |
| Yanlış veri kontrolleri | Shifted label64/64 reddi; ters feature, yanlış ölçek, NaN ve paydadan düşürme kontrolleri | diagnostic-tests-cpu.xml, c1_06r-regression.xml |
| İlk küçük öğrenme | local64 **FAIL 61/64**, mixed64 **PASS 64/64**, 5000 AdamW adımı | attempt-001/*-result.json ve epoch logları |
| Aynı objective LBFGS r2 | **FAIL 61/64**; 6 iterasyon/16 closure çağrısında durdu | diagnostic-r2/result.json |
| Aynı objective ×1e6, LBFGS r3 | **PASS 64/64**, 676 iterasyon/739 closure; aynı ilk checkpoint/model/örnek/dtype | diagnostic-r3/result.json |
| Ortam regresyonları | F0-01/02/03: 277; C1-02/03/04/05:177; yeni tanılar8 = **462 PASS**, skip0 | *-regression.xml (ayrı suite dosyaları) |
| Kesinti/devam ve bütünlük | **9 PASS**; dört arm gerçek CUDA son ağırlık hashleri kesintisiz koşuyla aynı | training-resume-tests.xml |
| Tam envanter kısa smoke | 4 arm ×2 epoch ×60 güncelleme =480 update; aynı başlangıç hashleri, validation/checkpoint/export PASS | training-smoke.json |
| Kullanıcı başlatıcısı | `pwsh -NoProfile -File scripts/Start-C106RTraining.ps1 -CheckOnly` exit0 | commands/018-launcher-check-only |

**471 ayrı test PASS** =462 regresyon/tanı+9 eğitim testi. Önceki CPU8 tekrarı
bu toplamda yeniden sayılmadı. GPU matematik ve gerçek overfit ayrıca raporlanır.
Tek pytest çağrısında farklı test klasörlerindeki aynı modül adları çakıştı:
`010` collection error5. Test adları/kabul eşikleri değiştirilmedi; suite'ler
ayrı süreçlerde çalıştırılarak düzeltildi. Hatalı collection ve iki başarısız
öğrenme tanısının kanıtları silinmedi.

Bu deneyde L-BFGS çözümleri inference sırasında sayısal IK refinement değildir:
yalnız **ağ parametreleri** küçük train kümesinde optimize edildi; her örnekte
doğrudan ağ q çıktısı bağımsız FK ile kontrol edildi. Ana eğitim yine yeni,
taze başlangıçlı AdamW koşusudur; tanı checkpointleri kullanılmaz. Loss ×1e6
yalnız r3 tanısında kullanıldı, ana eğitim kaybına sessiz taşınmadı.

## Sonuç ve yorum

Hedef-pose/teacher q, birim/frame, normalizasyon veya kopmuş gradyan kusuru
saptanmadı. Küçük sette doğru çözüm temsil edilebiliyor; ilk optimizer tanısında
yerel üç örnek başarısızdı. Pozitif Q-loss ölçeklemesi, aynı L-BFGS ve başlangıç
altında durma davranışını değiştirip Lq'yu yaklaşık1,57e−7'den1,92e−11'e
indirdi ve local64 A64/64 oldu. Bu **küçük tanı için ölçülmüş mekanizmadır**;
eski büyük validation/final sıfır başarısının tek nedeni diye genellenmez.

Eski C1-05 validation loglarında tanh seed1/3, Lq ile epoch7/8 seçilmişti;
etiketli p+R proxy'si epoch27/28'e kadar yaklaşık %44 daha düşük oldu.
Seed2'de fark yaklaşık %3'tü. Bu proxy Profil A değildir; sonradan eski
checkpoint seçilmedi ve eski final tekrar açılmadı. Kayıtlı batch'lerde tanh
saturasyonu düşük (en yüksek yaklaşık %0,684); bu tüm batch/popülasyonda
saturasyonun bulunmadığını kanıtlamaz. Kanıt: `r0r1/history-analysis.json`.

İlk uzun tur bu bulgular doğrultusunda **aynı mimari/veride** daha uzun,
cosine LR'lı eğitim ile poz başarısına dayalı seçimi sınar. FK ve başlık için
minimal 2×2 tasarım tek bileşen atfını mümkün kılar. Bu, %95'e ulaşıldığı veya
mutlaka ulaşılacağı iddiası değildir. Etiketli veri kapsamı genişletilmedi;
teacher eksikliği ve çoklu dal genellemesi açık araştırma başlıklarıdır.

Smoke eğitim çekirdeği ölçeğinden yaklaşık6 saat, I/O/validation/ısınmayla
yaklaşık6–11 saatlik kullanıcı kampanyası tahmini çıktı. Ölçülmüş tam kampanya
süresi değildir. GPU tensor peak yaklaşık71 MiB; küçük tanı peak süreç RAM
yaklaşık1,2 GiB. Uzun süreli kaynak/termal profil ve etkin insan emeği ölçülmedi.

## Sonraki adım

[USER_TRAINING.md](USER_TRAINING.md) tek hazır komutu verir. **Ana eğitim
NOT_RUN**, yeni final **NOT_CREATED**; SEALED denmedi. Kullanıcı koşuyu
başlatınca AI kayıt/hash/tamlık/validation analiziyle devam eder. Üç seedin
sonucu birlikte değerlendirilir; zayıf validation'da final açılmaz. Yeni final
ve yeni baseline kampanyası, test dağılımı/erişim ve H2-R kararları dondurulduktan
sonra ayrı yürütülür. C1-06 H2 REJECTED ve doğrudan IK NO-GO korunur;
collision/fiziksel güvenlik NOT_CHECKED, C1-07/G1 açık kalır.
