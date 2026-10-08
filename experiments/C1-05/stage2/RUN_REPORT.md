# C1-05 Aşama 2 çalışma ve kabul kaydı

Kimlik: RUN-20261008-C105-STAGE2

Durum: **COMPLETE / T-C04 PASS / E-C03, E-C04, E-C05 üç eşli seed tamam / doğrudan IK NO-GO**

Görev ve gereksinim: C1-05 / REQ-C03, REQ-C04

Tarih ve sorumlu: 8 Ekim 2026 · Codex; proje sahibi Arda Tekgöz.
Yazılım hedefi v1.0.0; belge r1. Stage1 dondurulmuş belgeleri ve config değişmedi.

## Soru ve değişiklik

Kullanıcının “aşama 2'ye geç” onayı `approval.json` içinde Stage1 commit ve
SHA256SUMS'a bağlandı. Lq kontrolüne FK terimleri, ayrı deneyde normalize ReLU
limit cezası, sonra ayrı deneyde tanh başlık eklendi. Quaternion/mutlak-q ve
3×256 SiLU backbone sabit; E-C06/07/08 ve Res-MLP/curriculum ön kayıtlı SKIP.

| Gereksinim | Değişiklik | Test | Ham kanıt / karar |
|---|---|---|---|
| REQ-C03 doğru eğitim FK | ADR-013 ayrı opt-in finite-revolute-extension, kamu FK kaynağı değişmedi | 80 iç/dış q, float64/32, 80 FD/gradcheck; 31 analitik/negatif | `domain/`, `physics-tests.xml`: PASS |
| REQ-C03 kayıp/graph/veri | Boyutsuz Lq/Lp/LR/Llim, sabit ölçek; label-only train | 5 training sözleşme testi, 19 kaynak mutant; 64/64 shifted label reddi | `training-tests.xml`, `mutations-v2/`, `pilot-assessment.json`: PASS |
| REQ-C04 yalıtılmış etki | Q/FK, FK/FK_LIMIT, FK/FK_TANH; fresh paired init/order/budget | 18 model-seed; 33210 optimizer step, 64800 validation kayıt | `E-C03/04/05`, `results-audit.json`: PASS |
| Regresyon | Eski C1-02/03/04 kaynağı korunur | 129 test,24 eski source mutant;1086 q/dtype ve32 gradient | `regression/`: PASS |
| Tekrar üretim/devir | Hashli best/last ve tüm validation q; temiz witness | 36 checkpoint byte/SHA,18 best×10 çıkarım ve FK | `results-audit.json`, `clean/`: PASS; fresh training NOT_RUN |

## Tekrar üretim

Stage1 başlangıç commit'i `9fe8bf34b9af039c651bb4dd60e3cf806948a23b`;
implementation `87faabf`, witness `45894c7fe3d8239afce9713c176f03e1019d8e14`.
Temiz checkout başlangıcı witness commit'indedir. Kapanış commit'i Git geçmişinde;
kendi hash manifestinin içine self-reference olarak yazılmaz. Kullanıcının
STATUS/TRACEABILITY denetim ekleri, PDF/DOCX/AUDIT dosyaları commit dışı korunur.

Windows11 x64 CPU; AMD Ryzen7 250, Torch2.10.0+cpu, Python3.12.14, NumPy2.5.3,
Pinocchio4.1.0; C1-03 iki hashli overlay lock, Foundations pixi.lock değişmedi.
Threads1/workers0, float32 eğitim; matematik FD float64. 11 exact artifact SHA
yeniden doğrulandı. Robot/TCP/frame/joint sözleşmesi ve C1-02 content SHA
`2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2` değişmedi.
Robot/config/normalizasyon/input/source SHA'ları checkpoint metadata ve
[devir manifestinde](C1-06-handoff.json). Model `.pt` dosyaları Git dışı
`data/generated/C1-05/{pilot,v1}/...`; yol/byte/SHA/erişim sonuç auditinde.

Eğitim seedleri2026100201–03, data seed2026092802, domain sample seed2026100805.
Train15204 labeled satır sabit;1596 unlabeled train satırı sadece
envanterde. Validation3600;3249 labeled ve351 unlabeled wide. Label olmayan
satırlar FK değerlendirmesinde tutulur. Her epoch q seçim metriği sadece3249
etiketli satırda float64 sum/count ile toplanan float32 loss'tur.

Pilot20 update/arm; ana18 model koşusunda33210 optimizer step. Ana eşli koşuların
eğitim+değerlendirme wall toplamı1496.236 s; driver/preflight/log overhead dahil
UTC aralığı `commands/s2-matrix` kaydındadır. Peak gözlenen RSS348291072 byte,
4 GiB tavanının altında. Etkin insan emeği ve inference latency NOT_MEASURED.
Temiz ortamda eğitim tekrarı, Linux/CUDA/fiziksel robot NOT_RUN.

## Test ve ham kanıt

Stage1 audit287 girdi/113 frozen çıktı PASS,34 shard ve6 eski checkpoint erişimi
yeniden doğrulandı. Yeni eğitim FK'sinde80 q×2 dtype ileri eşlik:
float64 max konum2.482534153247273e-16 m, R9.155133597044475e-16 (eşik1e-9);
float32 max konum1.5813959875927592e-7 m, R2.1378079558725061e-7 (eşik1e-5).
80 q'da16 objective×6 derivative, max FD fark2.1731527688473307e-9;
80 gradcheck PASS, h1e-6, atol1e-5/rtol1e-3. İç32 ve dış48 örneğin tümü test edildi.
NaN/Inf/shape/order/unit, extreme finite açı,1R/2R nonidentity base/TCP,
π çevresi, quaternion işareti, limit türevi ve tanh doygunluğu testleri geçti.

Pilot192 train/96 validation ve20 epoch, her bileşen için raw-output/katman
gradyanlarını, NaN/Inf,≥175° ve limit dışı oranı kaydetti. Başlangıç q/p/R raw-q
gradient normları0.03897/0.06069/0.03688; hepsi sonlu ve nonzero. FK'nin son
train Lq/Lp/LR'si0.7184/0.4796/0.5976. Ölçek açıklanamayan sapma göstermedi;
λq=λp=λR=1,ℓ0.9015 m korundu. Clamp/projeksiyon/drop-invalid yok.
`pilot-review.json` kararı; bütün ham q/FK-input/evaluation q ve kayıplar
`pilot/.../raw-pilot.jsonl`. Pilot ağırlıkları ana eğitime taşınmadı.

19 gerçek kaynak mutantı yakalandı; ilk denemedeki1 ERROR ayrı tutuldu,
yeniden değerlendirme19 KILLED/0 SURVIVED/0 ERROR. 129 eski regresyon ve24 eski
mutant PASS. Eski full T-C01/T-C02 de ayrı kökte yeniden PASS. Yeni model
validasyonu ayrı test sayılmaz: bütün64800 satır ve36 checkpoint sonuç auditinde
denetlendi. Q kontrolünün üç seed q çıktısı eski C1-04 ile **tam eş**.

Her koşu epoch loss (train/validation, mod, ağırlıklı/ağırlıksız), per-layer ve
raw-output bileşen gradient probe'u, her batch combined gradient/update normu,
permutation SHA, best/last kimlikleri ve3600 ham validation satırı üretir.
Ana koşu component gradient probe'u **epoch'un ilk etkin batch'i** içindir;
bütün veri gradient dağılımı diye sunulmaz. Tanh için boundary probe doygunluğu
seed1/2/3 maksimum0 /0.0375 /0; her eklem dq/dz ayrıca epoch JSONL'de.

Yeni boş checkout'ta `pixi install --locked`, fresh venv, iki hashli pip kurulum,
pip-check/exact-artifact ve18 best checkpoint×10 sabit inference/FK tekrarında
max fark0. Ortam kopyalanmadı; ağırlıklar özgün LOCAL_ONLY konumdan okundu.
`clean/start.json`,8 command logu ve `clean/witness-result.json` kanıttır.

## Sonuç ve yorum

[Tam sonuç tablosu](RESULTS.md), [paired farklar ve seed değişkenliği](results-audit.json).
**T-C04 PASS**, çünkü FK/limit/head etkileri ayrı, her ana kıyas üç seed,
arama ve gerçekleşen bütçeler kayıtlı, doğru FK/gradient ve hashli ham kanıt var.
Bu araştırma kabulü, başarılı doğrudan IK anlamına gelmez.

C1-04 conditioned yönelim medyanı74.6–78.1° iken E-C03 FK25.9–27.3° oldu.
Konum etkisi karma: +0.0091 /−0.0445 /−0.0403 m; ham limit ihlali arttı.
E-C04 limiti iki seed'de azalttı, birinde artırdı. E-C05 tanh ihlali sıfırladı.
**Bütün18 koşuda Profil A ve B0/3600; mutlak başarı iyileşmesi0.**
Dolayısıyla doğrudan IK NO-GO. Çarpışma/fiziksel güvenlik NOT_CHECKED.

E-C03 her seed200 epoch; E-C04 26/200/26, E-C05 27/200/28. Ön kayıtlı paired
patience20 koşulu tüm çiftlerde aynı şekilde çalıştı. Taze comparator'ın kısa
koşusu E-C03'ün200 epoch sonucu ile karıştırılmaz; iki arm her çift içinde aynı
bütçeyi aldı. Checkpoint q-loss ile seçildi, daha iyi FK epoch'una geçilmedi.
Bu erken durdurma etkisi ve düşük seed sayısı genelleme/üstünlük yorumunu sınırlar.

## Sonraki adım

Ön kayıtlı sıralama FK_TANH'ı yalnız **C1-06 araştırma adayı** seçer; üç seed
birlikte ve tanık seed2026100201 ile devredilir. [Model kararı](NEXT_MODEL_DECISION.md),
[hashli devir](C1-06-handoff.json),[acceptance](acceptance.json). Test/10000 sorgu
SEALED_NOT_RUN; C1-06 başlatılmadı. G1/v1.0.0 etiketi verilmedi. Uzak arşiv
NOT_CONFIRMED; gelecekteki çalışmadan önce checkpoint erişimi yeniden denetlenir.


## Gerçekleşen deney matrisi

| Deney | Durum | Gerekçe / bütçe |
|---|---|---|
| E-C03 Q/FK | RUN, 3 eşli seed | Matematik/pilot kapıları PASS; her seed 200 epoch |
| E-C04 FK/FK_LIMIT | RUN, 3 eşli seed | FK teknik kapısı PASS; 26/200/26 eşli epoch |
| E-C05 FK/FK_TANH | RUN, 3 eşli seed | E-C03 FK'de limit ihlali vardı; 27/200/28 eşli epoch |
| E-C06 quaternion/6D | SKIP | Ön kayıtlı ilk dalgada ek temsil bütçesi ayrılmadı |
| E-C07 mutlak/delta q | SKIP | Mutlak q sabit; delta yararı ölçülmedi |
| E-C08 tekillik cezası | SKIP | Önce σmin tanısı; ikinci dalga önkoşulu |
| Res-MLP/curriculum | SKIP | Ayrı ADR ve ön kayıt gerekir |

Dört özgün config, her arm için bir aday; adaptif arama yapılmadı. Sekiz config
tavanı doldurulmadı. Stage1 audit çıktısındaki eğitim/T-C04 NOT_RUN alanları
dondurulmuş Aşama 1 durumudur; Aşama 2 kararı `acceptance.json` içindedir.

Kanıt dosyalarının içerik manifesti `evidence-manifest.json` ve `SHA256SUMS`
içindedir. Temiz checkout'ta bu manifestin son denetimi ayrıca
`postcommit/clean-bundle-audit.json` ile kaydedilir; kendi sonucunu hashleme
döngüsünü önlemek için bu sonraki kayıt frozen snapshot dışındadır.
