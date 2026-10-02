# C1-04 Aşama 1 çalışma kaydı

Kimlik: RUN-20261002-C104-STAGE1

Durum: **IN_PROGRESS / STAGE_1_COMPLETE; T-C03 NOT_RUN; E-C01 NOT_RUN**

Görev ve gereksinim: C1-04 / REQ-C03

Tarih ve sorumlu: 2 Ekim 2026 · Codex; proje sahibi Arda Tekgöz

Yazılım hedefi: v1.0.0; belge revizyonu: C1-04 r1, STATUS r19, roadmap r10

## Soru ve değişiklik

G0, C1-02 ve C1-03 kabulünden sonra iki neural baseline'ın adil E-C01 karşılaştırması ve T-C03 öğrenme kapısı, herhangi bir eğitim sonucu görülmeden dondurulabilir mi? Evet: [tasarım incelemesi](STAGE1_REVIEW.md), [config](config.json), [checkpoint şeması](checkpoint-schema.json), [test matrisi](TEST_MATRIX.md), [negatif kontrol matrisi](NEGATIVE_CONTROL_MATRIX.md) ve [komut planı](COMMANDS.md) hazırlandı. Önceki görevlerin eşikleri/varlıkları değişmedi. Model/loader kodu, eğitim, checkpoint ve neural metrik bu aşamada üretilmedi.

Gereksinim → değişiklik → test → kanıt:

| Gereksinim | Aşama 1 değişikliği | Gerçek kontrol | Kanıt/karar |
|---|---|---|---|
| Değişmez robot, TCP, FK ve veri girdileri | SHA-256 kaynak/yerel shard manifesti; kimlik sözleşmesi | 62 dosya SHA/erişim ve kabul metadata audit'i; C1-03 kanıt manifesti | [input-hashes.json](input-hashes.json), [stage1-commands.json](stage1-commands.json); PASS |
| E-C01 adil iki model | 7/13 özellik, aynı etiketli satırlar ve sıra, üç seed, loss/bütçe/seçim | JSON sözdizimi ve parametre sayısı hesap audit'i | [config](config.json), [inceleme](STAGE1_REVIEW.md); eğitim NOT_RUN |
| T-C03 gerçek küçük öğrenme ve yanlış eşleme | Ön kayıtlı 64 train/32 validation; pozitif/negatif eşikler | Sadece test tasarımı; eğitim/negatif mutasyon henüz koşulmadı | [test matrisi](TEST_MATRIX.md); T-C03 NOT_RUN |
| İzlenebilirlik ve devir | Görev, roadmap, STATUS, TRACEABILITY tarihli ekleri | Git diff/hash kontrolü | Bu rapor ve ilgili kayıtlar; C1-05 NOT_STARTED |

## Tekrar üretim

Başlangıç `main`, yerel `origin/main` ve uzak `refs/heads/main`: `93146c82108a60462b3108d799ba83e610a8bfef`. Başlangıç çalışma ağacında üç ilgisiz izlenmeyen dosya (`NeuroKinematics_10.08.2026.pdf`, `raporlar/01_Foundations_v0_1_sonuc_r2.docx`, `~$uroKinematics_Model_Kullanim_Plani_r1.docx`) vardı; kapsam dışı tutuldu. Stage1 kapanış commit'i bu rapora self-reference eklememek için Git geçmişi ve final teslimde verilecektir.

Gerçek host Windows 11 Pro 10.0.26200, AMD Ryzen 7 250, fiziksel RAM 25.025.695.744 byte. `pixi.lock` ve C1-03 iki exact wheel lock'u [girdi manifestinde](input-hashes.json). C1-03 overlay'de Python 3.12.14, Torch 2.10.0+cpu, NumPy 2.5.3, Pinocchio 4.1.0 importu çalıştı. Bu yalnız ortam kontrolüdür; C1-04 eğitimi veya Linux/CUDA doğrulaması değildir. Planlanan eğitim tek CPU thread, sıfır worker. Stage1 hash audit'i host Python 3.11 standard library ile de çalışır.

Robot `kuka_kr6_r900_sixx`, `base_link`→`tool0`, `joint_1`–`joint_6`, rad/m; robot URDF `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96`, manifest `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1`, TCP `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18`. C1-02 içerik SHA `2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2`; 34 yerel shard toplam **15.598.536 byte**; dataset kökü 39 dosya/**44.479.998 byte** (teacher aday dosyası dahil). `data/generated/C1-02/v1` ve `v1-repro` `LOCAL_ONLY`; uzak arşiv `NOT_CONFIRMED`. Etiketli train/validation 15.204/3.249; 2.281 etiketsiz wide satır korunur. Veri seed'i `2026092802`; frozen eğitim seedleri [config](config.json) içinde `2026100201`–`03`.

Aşama 1 gerçek komut/exit kaydı [stage1-commands.json](stage1-commands.json) içinde; tekrar komutu `python scripts/check_c104_stage1.py --check`. Dondurulan Stage1 dosyalarının SHA-256 listesi [SHA256SUMS](SHA256SUMS). Bu aşamada model/checkpoint SHA'sı **YOK**, eğitim başlangıç/bitiş zamanı ve eğitim süresi **NOT_MEASURED**, etkin insan emeği **NOT_MEASURED**.

## Test ve ham kanıt

| Kontrol | Gerçek sonuç | Karar |
|---|---|---|
| `python scripts/check_c104_stage1.py --write-manifest` | 62 girdi, 34 shard, 15.598.536 shard byte; C1-02/C1-03 kabul metadata eş | exit 0 / PASS; yalnız Stage1 dondurma |
| `python scripts/check_c104_stage1.py --check` | Dondurulan 62 dosya SHA eş | exit 0 / PASS |
| `python scripts/manifest_c103.py` | 464 dosya manifest audit | exit 0 / PASS |
| C1-03 overlay import/version | Python 3.12.14, Torch 2.10.0+cpu, NumPy 2.5.3, Pinocchio 4.1.0 | exit 0 / PASS; matematik yeniden koşulmadı |
| Eski `python scripts/check_c102_stage1.py --check` | Yalnız `.gitattributes` eski dondurma SHA'sından farklı; C1-03 ve C1-04 için sonraki Git metin kuralları eklendi. C1-02 config/schema/veri shard sapması yok. | exit 1 / tarihsel checker kapsam drift'i; gizlenmedi |
| C1-02 canlı `pair_validation.verify` | 24.000 satır, 34 shard, canonical SHA eş; leakage, soy, train-only normalizasyon ve bağımsız etiket FK PASS | exit 0 / [ham sonuç](c102-readonly-verify.json); C1-04 eğitimi değildir |
| T-C03 / E-C01 / neural FK ve checkpoint | Koşulmadı | NOT_RUN / NOT_MEASURED |

Eski C1-02 Stage1 checker'ın başarısız sonucu kabulü geriye dönük iptal etmez; sebep [stage1-commands.json](stage1-commands.json) içindeki eski/yeni `.gitattributes` hash çiftidir. Canlı veri denetimi C1-02 üretim kodunun `verify` fonksiyonunu doğrudan çağırır; kapsamı veri/etiket/soy/normalizasyondur. C1-04 model doğruluğu değildir. Eğitimden önce Stage1 `--check` yeniden geçmelidir.

## Sonuç ve yorum

REQ-C03 için C1-04 sözleşmesi denetlenebilir biçimde hazır; **T-C03 PASS veya C1-04 COMPLETE kararı verilmedi**. Pose-only aynı hedef pozu farklı current/teacher etiketiyle görebilir; bu E-C01'in ölçülmesi gereken yapısal sınırlamasıdır. Ham 50/50 ile etiketli train dağılımının farklılığı saklanmadı. Test ve benchmark model seçimine kapalı. Geçersiz ham q gelecekte sessiz kırpılmayacak. Sayısal IK başarısızlığı erişilemezlik kanıtı değildir; FK metriği çarpışmasızlık/fiziksel robot güvenliği göstermez.

## Sonraki adım

Aşama 1 commit/push ve SHA tesliminden sonra açık kullanıcı onayı bekle. Onay gelirse önce Stage1 SHA audit; ardından yükleyici/iki MLP/fail-fast, T-C03 pozitif-negatif kapısı ve ancak o geçerse altı E-C01 koşusu. C1-05 devri için model/checkpoint/config/scaler/hash erişim paketi henüz yoktur. Linux/CUDA/fiziksel robot NOT_RUN, G1/v1.0.0 kapanışı yok.
