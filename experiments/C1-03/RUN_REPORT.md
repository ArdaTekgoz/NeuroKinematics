# C1-03 Aşama 1 çalışma kaydı

Kimlik: RUN-20260929-C103-STAGE1
Durum: **IN_PROGRESS / STAGE_1_COMPLETE**
Görev ve gereksinim: C1-03 / REQ-C02; T-C01, T-C02 **NOT_RUN**
Tarih ve sorumlu: 29 Eylül 2026 · Codex; proje sahibi Arda Tekgöz
Yazılım hedefi: v1.0.0 · Belge revizyonu: r1

## Soru ve değişiklik

G0 hashli robot/frame zincirini ve bağımsız referansı koruyarak Torch FK ve
gradyan kabulünün sözleşmesi donduruldu. [İnceleme](STAGE1_REVIEW.md),
[config](config.json), [örnekleme](SAMPLING_CONTRACT.md), [test matrisi](TEST_MATRIX.md),
[negatif/mutasyon matrisi](NEGATIVE_MUTATION_MATRIX.md), [girdi manifesti](input-hashes.json),
[bağımlılık pinleri](dependency-pins.json), [SHA256SUMS](SHA256SUMS), statik checker
ve [ADR-011](../../docs/adr/ADR-011-c103-torch-fk.md) eklendi. Üretim FK kodu değişmedi.

`pytorch-kinematics==0.10.0` MIT/API/wheel kimliği doğrulandı. Seçim mevcut parser
üzerinde `native-torch-serial-v1`; standart autograd ve fixed/revolute kapsamı.
Torch 2.10.0+cpu ayrı hashli Windows overlay için seçildi; kurulum **NOT_RUN**.
Referans Pinocchio 4.1.0, robot/TCP, eski FK/Jacobian ve Pixi lock korunur.

## Tekrar üretim

Başlangıç main / origin/main: `b82faf95ef87666471fd5ae77ec1d392364a108a`.
İlgisiz iki untracked Word dosyası commit dışı. Ortam Windows 11 x64,
Pixi 0.81.0, CPython 3.12.14, NumPy 2.5.3, Pinocchio 4.1.0, pytest 8.4.2;
[environment.json](environment.json). CPU AMD64 Family 25 Model 117 Stepping 2.
RAM/GPU performansı ve insan etkin emeği **NOT_MEASURED**; GPU kullanılmadı.
Stage1 yalnız seri statik kontrol; BLAS thread ölçümü yapılmadı. Stage2 tek thread CPU.
Başlangıç/bitiş tam oturum saati ölçülmedi; kaydedilmiş komut UTC aralıkları
[commands.json](commands.json) içindedir ve etkin emek yerine geçmez.

| Kimlik | SHA-256 |
|---|---|
| URDF | 83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96 |
| RobotSpec | 4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d |
| Robot manifest | aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1 |
| TCP kuka_tool0_zero | 52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18 |
| Foundations pixi.lock | 56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a |
| C1-03 config r1 | f7064ba8ee1587aa90fbb04efc3c1899e381b2a19a9cf01d6b7b2e095e664d6d |
| samples.jsonl (1086) | 085fdda74dbf89b9098f6a6a0627576eb2302fb705dd379288c9ff7a187e03dc |
| Torch cp312 win_amd64 wheel (index hash; runtime NOT_RUN) | 21cb5436978ef47c823b7a813ff0f8c2892e266cfe0f1d944879b5fba81bf4e1 |

PCG64 FK seed 2026092903, gradient seed 2026092904. Eğitim seed/model/checkpoint
**YOK**. Veri split'i uygulanamaz; C1-02 raw shardları kullanılmadı.
Yeni dosyalar UTF-8/LF, checksum canonical LF; geçmiş raw/Git blob içerik SHA'ları
ayrı tutulur. Tam tarif [COMMANDS](COMMANDS.md). Geçici metadata scriptindeki
eksik index hash ve Windows default encoding hataları gizlenmedi; düzeltme
COMMANDS'ta. FK veya kabul eşiği bu düzeltmelerden etkilenmedi.
İlk diff-check üç Markdown satır sonu boşluğunu yakaladı; boşluklar kaldırıldı,
başarısız loglar `logs/diff-check-first.*.log` olarak korundu.

## Test ve ham kanıt

| Gereksinim | Değişiklik | Gerçek kontrol ve sonuç | Kanıt |
|---|---|---|---|
| Frozen robot/G0 | Girdi manifesti, statik checker | 15/15 devir, 29/29 robot dosyası raw hash eş; 85 girdi/reference envanteri | input-hashes.json, stage1-check.json |
| Joint/frame/limit | Matematik sözleşmesi | 6 revolute + 2 fixed yol, sıra/limit yapı denetimi | stage1-check.json chain |
| Örnekleme/config | 1086 q ve 32 ayrı gradient q; iki dtype exact baytları | Shape/finite/limits/benzersizlik ve deterministik üretim kontrolü | samples.jsonl, stage1-check.json |
| Paket/API/lisans | Wheel metadata, Windows CPU hashli overlay | Aday wheel SHA ve kaynak incelemesi; pip dry-run exit0 | candidate-metadata.json, dependency-resolution.json/log |
| T-C01 f64/f32 | Frozen tüm-örnek toleransları | **NOT_RUN**, maksimum hatalar **NOT_MEASURED** | TEST_MATRIX.md, config.json |
| T-C02 | 32 q / 2880 FD bileşeni planı | **NOT_RUN**, en kötü türev farkı **NOT_MEASURED** | SAMPLING_CONTRACT.md |
| Mutasyon/negatif | 24 hata sınıfı planı | **NOT_RUN**, 0 yürütülen C1-03 mutation testi | NEGATIVE_MUTATION_MATRIX.md |
| Regresyon/temiz tekrar | F0-01/02/03 + C1-02 fixture ve ikinci fresh kurulum | **NOT_RUN**, 0 yürütülen regression testi | TEST_MATRIX.md |

Eski G0/F0/C1-02 PASS kayıtları bu oturumda yeniden çalıştırılmış test sayılmaz.
Yalnız hash, yapı, örnekleme ve tasarım doğrulandı; `PASS_STATIC_ONLY` görev
PASS'ı değildir. Yeni JUnit yok çünkü henüz sayısal/test suite koşusu yapılmadı.

## Sonuç ve yorum

Stage1 sözleşmesi tamamlandı; C1-03 kabulü verilmedi. Başlıca açık konular:
Torch'un bu Pinocchio/Windows DLL ortamıyla gerçek kurulum/import uyumu,
float32 hataları, p/R autograd doğruluğu, tüm mutantların öldürülmesi ve ikinci
temiz koşu. Hepsi Aşama2 kapısında çözülmelidir; ölçülmeyen sonuç başarılı sayılmaz.
Tekillik adaylarının sigma_min'i NOT_MEASURED; sonuç temelli örnek seçilmedi.
Quaternion işaret/pi süreksizliği matrix FK türevinden ayrı tutulur.
Fiziksel doğruluk/collision/robot güvenliği bu çalışma ile doğrulanmaz.

Küçük kanıt ve örnek dosyaları Git kapsamındadır; yeni büyük raw/model yok.
C1-02 dosyaları ve kullanıcı değişiklikleri korunmuştur. Bu kaydı ekleyen commit
Git geçmişinden okunur; push ve local/remote SHA eşitliği committen sonra doğrulanır.
v1.0 etiketi ve G1 kararı oluşturulmaz.

## Sonraki adım

Burada dur ve kullanıcıdan **açık Aşama 2 onayı** iste. Onay sonrası ayrı approval
kaydı ve Stage1 SHA audit, ardından küçük forward/analitik/smoke sırası uygulanır.
Tam kabul, matrisin bütün zorunlu kontrolleri ve ikinci temiz kurulumla kapanır.
C1-04 başlatılmaz; C1-02 kabulüne ek olarak C1-03 kabulünü de gerektirir.
