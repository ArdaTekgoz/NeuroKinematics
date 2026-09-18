# F0-02 bağımsız ileri kinematik doğrulama kaydı

Kimlik: RUN-20260918-003
Durum: PASS · KABUL · TAMAMLANDI
Görev ve gereksinim: F0-02 · REQ-F02 · T-F02
Tarih ve sorumlu: 18 Eylül 2026 · Codex uygulaması; proje sahibi Arda Tekgöz
Yazılım hedefi: v0.1.0 · Belge revizyonu: r1

## Soru ve değişiklik

Aynı immutable URDF için bağımsız XML/NumPy seri zincir FK, Pinocchio 4.1.0
referansıyla `T_base_link_tool0(q)` üzerinde uyuşuyor mu? Önceden dondurulmuş
kabul: 10.000 geçerli float64 q; maksimum konum L2 farkı ≤ **1e-9 m** ve
maksimum dönme matrisi Frobenius farkı ≤ **1e-9** birlikte. Seed, örnekleme
yöntemi, immutable girdiler veya bu eşikler sonuç görüldükten sonra değiştirilmedi.

`model.py` raw bayt hashlerini JSON/XML parse öncesinde doğrular ve aynı baytları
adaptörlere verir. `chain.py` ElementTree ile tek yönlü yolu kendisi çıkarır;
`custom_fk.py` yalnız bu zincirdeki NumPy float64 dönüşümlerini çarpar.
Pinocchio modeli, origin/axis bilgisi, frame placement veya ara sonuçları bağımsız
hesapta kullanılmaz. Ortak kod yalnız girdi kimliği ve q/limit doğrulamasıdır.
Pinocchio importunu yasaklayan ayrı Python sürecinde bağımsız FK çalıştırıldı.
Markaya özel geometri veya DH tablosu eklenmedi; veri/eğitim/GUI/solver bağımlılığı yok.

URDF sabit eksen RPY: `Rz(yaw) @ Ry(pitch) @ Rx(roll)`. Sütun vektörleri,
metre/radyan, sağ el kuralı ve joint-local axis kullanılır. Her adım
`T_origin @ T_motion` olarak uygulanır. Altı aktif ekleme ek olarak
`joint_6-flange` ve `flange-tool0` korunur; `base_link-base` yan dalı dışarıda kalır.
Joint vektörü XML sırasından bağımsız olarak manifest sırasındadır.

`pinocchio_fk.py` immutable XML baytlarından model kurar, joint adlarını `idx_q`
ile eşler, BODY frame kimliklerini açıkça çözer, `forwardKinematics` ve
`updateFramePlacements` çağırır. Sonuç `inverse(T_world_base) @ T_world_tool0`
ile hesaplanır. Gerçek modelin base identity olduğu ve identity olmayan sentetik
base için relatif dönüşümün doğru olduğu ayrı testlerle doğrulandı.

## Tekrar üretim

| Alan | Gerçek değer |
|---|---|
| Başlangıç HEAD | `1e0131122047aa8992601c7bb6e0cc94525bd709` |
| Uygulama commit'i | `d92dd213bb96f8932bd0019541dd13dd7365afaf` |
| Başlangıç ağacı | `main...origin/main`, ahead/behind `0/0`; yalnız kullanıcıya ait izlenmeyen `~$uroKinematics_Model_Kullanim_Plani_r1.docx`; dokunulmadı |
| Ön koşul | `git fetch origin`; F0-01 uygulama `4048c428afceaab4418d6107897dcd36c2d48f33` ve kapanış `1e0131122047aa8992601c7bb6e0cc94525bd709` origin/main ataları |
| Kaynak commit | `fbda927964caa1eb4e408fb0c25fe46b5a0bde3c` |
| Robot / frames | Standart KUKA KR 6 R900 sixx; `base_link` / `flange` / `tool0` |
| Ortam | Windows 11 10.0.26200 AMD64; Python 3.12.14; Pixi 0.81.0 |
| Paketler | NumPy 2.5.3; Pinocchio 4.1.0; pytest 8.4.2; Xacro 2.1.1 |
| CPU / RAM / GPU | AMD64 Family 25 Model 117 Stepping 2; 16 logical CPU; RAM NOT_MEASURED; GPU NOT_USED |
| Thread / runtime | Seri CPU döngüsü, tek pytest süreci; BLAS thread çevre değişkenleri UNSET; yerel BLAS thread sayısı ölçülmedi |
| Lock | `pixi.lock` SHA-256 `56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a`; yeni bağımlılık veya lock farkı yok |
| Config | [`config.json`](config.json); seed `20260918`; NumPy Generator / PCG64 |
| Örnekleme | Immutable limitler arasında `Generator.uniform(lower_rad, upper_rad, size=(10000,6))` |
| Sayısal tür | `(10000,6)` float64; bütün q ve dönüşümler sonlu ve geçerli |
| Sample SHA-256 | `8fb7e88758aa841310ae4d665d76d00a4488a5b79217ca4d5c80a825715c7101` |
| Hash biçimi | C-order little-endian float64 raw baytlar; shape ayrıca kayıtlı; `.npy` metadata'sı hashlenmez |
| Tam kapanış komutu | `pixi run --locked python scripts/run_f02_acceptance.py` |
| Frozen koşu zamanı | 18 Eylül 2026 21:43:27–21:43:43 Europe/Istanbul; exact UTC zamanları `commands.json` içinde |
| Tam oturum emek süresi | Ölçülmedi; yukarıdaki süre yalnız kabul komutlarının duvar saati aralığıdır, FK benchmarkı değildir |
| Veri/split/model/checkpoint | Uygulanamaz; veri fabrikası ve öğrenme başlatılmadı |

Immutable hashler başlangıçta, testlerde ve commit öncesinde yeniden doğrulandı:

| Varlık | SHA-256 |
|---|---|
| `assets/robots/robot_a/robot.urdf` | `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96` |
| `assets/robots/robot_a/robot_spec.json` | `4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d` |
| `assets/robots/robot_a/manifest.json` | `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1` |
| `config/robots/tcp_tool0.json` | `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18` |

## Test ve ham kanıt

| Kontrol | Ölçülen sonuç | Karar / kanıt |
|---|---|---|
| Frozen lock/install | İkisi de exit 0 | PASS; [`commands.json`](commands.json) |
| T-F00 regresyon | 6/6; 0.37 s | PASS; [`f00-junit.xml`](f00-junit.xml) |
| F0-01 varlık kontrolü | 7 kritik kaynak, 29 dağıtılan dosya, 14 mesh, nq=nv=6 | PASS; `commands.json` |
| T-F01 regresyon | 16/16; 0.68 s | PASS; [`f01-junit.xml`](f01-junit.xml) |
| F0-02 unit/regresyon | 101/101; 1.49 s; 1000-q regresyon dahil | PASS; [`unit-junit.xml`](unit-junit.xml) |
| Nihai CLI doğrulaması | 10000/10000 geçerli karşılaştırma | PASS; [`fk-validation-summary.json`](fk-validation-summary.json) |
| T-F02 tam pytest | 102/102; 5.14 s; 10000-q test gerçekten çalıştı | PASS; [`pytest-junit.xml`](pytest-junit.xml) |
| Tekrar üretim | CLI ve ayrı pytest hesabında aynı sample hash ve maksimum hatalar; ikinci tam koşu da aynı sonuç | PASS |
| Kanıt bütünlüğü | SHA256SUMS içindeki 37 dosya raw bayt hashleriyle eşleşti | PASS; [`SHA256SUMS`](SHA256SUMS) |

T-F02 dağılımları (percentile yöntemi NumPy varsayılan `linear`):

| Metrik | Min | Median | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Konum L2, m | 0 | 5.551115123125783e-17 | 2.220446049250313e-16 | 2.498001805406602e-16 | **4.75098925995612e-16** |
| Rotation Frobenius | 1.942890293094024e-16 | 5.36049202688126e-16 | 7.012844585452197e-16 | 7.81618876384399e-16 | **9.159602786276758e-16** |

Eşik aşan örnek: **0**. Sonlu olmayan sonuç: **0**. Yapısal olarak geçersiz sonuç:
**0**. Eşiğin %90'ına ulaşan örnek: **0**. Boş hata listeleri
[`diagnostics.json`](diagnostics.json) içinde açıkça tutulur. Maksimum konum
örneği 8784, maksimum rotation örneği 4651; sıfır tabanlı index, q, iki translation,
iki hata ve en büyük mutlak matris elemanı farkı özet JSON'dadır.

21 elle seçilmiş konfigürasyon; sıfır, orta noktalar, her joint için ±0.25 rad,
alt/üst limit yakını, doğrudan alt/üst limit ve üç karışık işaretli q içerir.
Her birinin bağımsız ve referans 4×4 matrisleri ayrı ayrı
[`handpicked-results.json`](handpicked-results.json) içindedir.

| Gereksinim / hata sınıfı | Uygulama | Otomatik kanıt |
|---|---|---|
| Analitik dönme, RPY, origin-motion, yerel axis | `transforms.py`, `custom_fk.py` | `test_math_chain.py` |
| Tek yol, fixed/TCP, aktifler arasında fixed, desteklenmeyen joint/graph | `chain.py` | `test_math_chain.py`, `test_robot_reference.py` |
| Joint adı/sırası, base/frame, flange≠tool0 | İki ayrı FK adaptörü | `test_robot_reference.py` |
| NaN/Inf/boyut/birim/limit ve immutable hash reddi | `model.py` | `test_robot_reference.py` |
| Bağımsızlık | Pinocchio'suz ayrı süreç; kod bağımlılık sınırı | `test_custom_fk_without_pinocchio_import_or_oracle` |
| Örnek hash'i, başarısız/sonlu olmayan/eşik yakın kayıt, yalnız rotation hatası | `validation.py` | `test_evidence.py` |
| İki maksimumla nihai 10000-q kapısı | `validation.py` | `test_acceptance.py`, JUnit properties |

İlk başarısız denemeler ve düzeltmeler:

1. Identity olmayan sentetik base için `SE3.inverse()` ile analitik testte
   `2.6645352591003757e-15 > 2e-15` fark görüldü (97 PASS / 1 FAIL).
   Dünya ötelemesini geçici küçültme de `2.220446049250313e-15` ile başarısız oldu;
   bu değişiklik geri alındı. Özgün `[3,4,5]` ötelemesi, dönmüş base ve `2e-15`
   test sınırı korundu. Referans adaptörü, SE3'ün tam ortonormallik varsayımıyla
   R-transpose kullanması yerine temsil edilen float64 base matrisini açıkça
   tersliyor. Özgün analitik test bu düzeltmeyle geçti.
2. İlk smoke CLI, conda Pinocchio için `importlib.metadata` kaydı bulamadı;
   ortam raporu tamamlanmadı ve exit 1 verdi. Sürüm `pinocchio.__version__` ile
   okunuyor. Kabul kararı destek dosyaları tamamlandıktan sonra yazılıyor.
3. İlk toplu runner, başarılı T-F00 çıktısını cp1254 stdout'a basarken durdu.
   Runner ve alt süreç Python I/O'su UTF-8 yapıldı; kesilen komutların ham kaydı
   [`initial-interrupted-commands.json`](initial-interrupted-commands.json) içinde.
4. İlk tam koşu 99 testle geçti. Son kapsam kontrolünde aktifler arasında fixed,
   yalnız rotation hatası ve eşik yakını kayıt testleri eklendi; örnek hash'i
   golden assertion ile bağlandı. Son frozen koşu 102/102 geçti.

Deneme sonuçları ayrıca [`development-attempts.json`](development-attempts.json)
içinde, araç çıktılarından aktarılmış özet olarak açıkça etiketlidir. Nihai ham
komut çıktıları [`commands.json`](commands.json) içindedir. Başarısız denemeler
kabul sayılmadı; nihai seed, örnekler ve kabul eşikleri hiç değiştirilmedi.

## Sonuç ve yorum

F0-02 **PASS / KABUL / TAMAMLANDI**. İki maksimum da önceden belirlenen sınırların
altında; immutable girdiler, analitik testler, bağımsızlık, hata yolları ve
F0-00/F0-01 regresyonları birlikte geçti. 1000-q smoke raporu tek başına kapanış
olamayacağı için `INCONCLUSIVE` olarak korunmuştur.

Açık sınırlamalar:

- Yalnız Windows yürütüldü; Linux **NOT_RUN**. Paket kilidi değişmedi.
- İki hesap aynı URDF'yi kullanır; yanlış ortak fiziksel model bu karşılaştırmayla
  dışlanamaz. Fiziksel kalibrasyon, collision, dinamik ve robot güvenliği doğrulanmadı.
- F0-01'den kalan üretici PDF bayt/hash eksikliği devam ediyor.
- Pinocchio adaptörü mutable Data tutar; eşzamanlı kullanım için ayrı instance gerekir.
- API q değerlerini radyan olarak yorumlar. Limit içindeki yanlış birim veya yanlış
  sıra niyetini çıplak sayılardan tespit edemez; bunları sözleşme ve regresyonlar sınar.
- Gerçek `joint_6-flange` ve `base_link-base` dönüşümleri identity olduğu için yalnız
  sayısal fark onları saptayamaz; yapısal ve nonidentity sentetik kontroller birlikte kullanıldı.
- Hız/gerçek zaman garantisi, Jacobian, IK, veri fabrikası ve öğrenme **NOT_RUN**.
- SHA256SUMS kendisini ve kapanış Markdown anlatısını dışlar; uygulama kanıtları,
  kaynak/test dosyaları, Pixi manifest/lock ve runner'ı kapsar. Ham 10000 satırlık
  q matrisi Git'e eklenmedi; config/seed/hash ile yeniden üretilir.

## Sonraki adım

**F0-03'e geçiş hazırdır; F0-03 başlatılmadı.** Foundations fazı/G0 kapanmadı.
Devredilen API:

```python
from neurokinematics.kinematics import IndependentFK, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK

inputs = load_robot()
custom = IndependentFK(inputs)
reference = PinocchioFK(inputs)
T_base_tool0 = custom.forward_kinematics(q)
T_base_tool0_reference = reference.reference_forward_kinematics(q)
```

İki çıktı `(4,4)` float64, metre/radyan, sütun-vektör sözleşmesindedir.
F0-03 immutable dört girdi hashini, bu uygulama commit'ini, `config.json`,
`sample-hash.json`, `fk-validation-summary.json` ve `SHA256SUMS` içindeki kaynak
hashlerini devralır. Jacobian doğrulamasını ayrı görev olarak yürütmelidir.

İkinci commit bu rapor, komut anlatısı, görev/durum/roadmap/izlenebilirlik ve
kurulum güncellemelerini taşır. Kendi SHA'sını içine yazmak için üçüncü commit
üretilmez; kapanış SHA'sı ve gerçek normal push sonucu son kullanıcı yanıtında
ve Git remote kaydında bildirilir.
