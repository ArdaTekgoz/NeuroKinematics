# C1-03 Aşama 2 çalışma kaydı

Kimlik: RUN-20260929-C103-STAGE2
Durum: COMPLETE / PASS / ACCEPTED
Görev ve gereksinim: C1-03 / REQ-C02 / T-C01 ve T-C02 PASS
Tarih ve sorumlu: 29 Eylül 2026 · Codex; proje sahibi Arda Tekgöz
Yazılım hedefi: v1.0.0; belge r2. Aşama 1 belgeleri değişmez.

## Soru ve değişiklik

Kullanıcının açık “Onaylıyorum” yanıtı [approval.json](approval.json) ile Stage1
commit ve manifestine bağlandı. `native-torch-serial-v1` yaklaşımı doğrulanmış
URDF zincirini standart Torch fixed/revolute kernel ile hesaplar.
[torch_fk.py](../../../src/neurokinematics/kinematics/torch_fk.py) tek `(6,)`
veya batch `(N,6)` q için `T_base_tcp` üretir; dtype/device ve autograd grafiğini
korur. Forward hesabında NumPy, detach, item, no_grad veya Pinocchio yoktur.
Kimlik, sıra, birim, limit, shape ve nonfinite hataları açık reddedilir.
Pinocchio yalnız [validator](../../../src/neurokinematics/core/torch_validation.py)
içinde bağımsız sayısal referanstır. Foundations/C1-02 kaynakları ve kanıtları
korundu. Büyük C1-02 shardları veya teacher etiketi gerekmez.

Robot `kuka_kr6_r900_sixx`; `joint_1`–`joint_6` tam sırası, radyan/metre,
kolon vektör ve sağ el kuralı; `base_link` → `tool0`. Sabit TCP `Ry(π/2)`
özdeşlik değildir. Ayrı analitik fixture'lar nonidentity base/TCP, origin/axis
ve 180° rotasyonu denetler. Quaternion FK çıktısı değildir; işaret/π köşeleri
ayrı analitik testtedir.

## Tekrar üretim

Stage1 başlangıç commit'i `6a56e9e9eeaabb561cb3efdc899a5e5709876783`;
uygulama `7d9e282`, exact overlay audit düzeltmesi
`4022e2359306a780422c94f25252bd2eaa90ed8f`. Birincil kabul kanıtı
`full-exact-a/`; ikinci gerçek temiz checkout/kurulum `clean-b/`.
[clean-start](clean-b/clean-start.json) başlangıçta temiz Git ve bulunmayan
`.pixi`/`.venv` kaydeder; ortam dosyası kopyalanmadı. İkinci koşunun kaynak
commit'i `4022e2359306a780422c94f25252bd2eaa90ed8f`;
[complete](clean-b/complete.json) 15/15 komut PASS. Dört BLAS/OpenMP değişkeni
ve Torch thread sayısı 1. Windows 11 x64 CPU; Python 3.12.14, Torch 2.10.0+cpu,
NumPy 2.5.3, Pinocchio 4.1.0, pytest 8.4.2. Paket yolları/sürümleri
[environment](full-exact-a/environment.json), exact 11 wheel URL/SHA kimliği
[artifact audit](commands/024-artifact-audit/stdout.log) içindedir.

Kapanış sırasında aynı hosttan okunan donanım: AMD Ryzen 7 250 (8 core/16 thread),
25.025.695.744 bayt fiziksel RAM; NVIDIA GeForce RTX 5060 Laptop GPU ve AMD Radeon
780M Graphics. GPU kullanılmadı. Bu donanım kaydı performans benchmark'ı değildir.
Seeds: FK `2026092903`, gradient `2026092904`. 1024 random +32 gradient +30
hand/edge =1086 q/dtype; örnekler Stage1'de sonuçlardan önce donduruldu.

| Kimlik | SHA-256 |
|---|---|
| Stage1 SHA256SUMS (canonical LF) | `0ee8682e7848225a4e173519760d7cfbbf04aca41172a2938fbcd0b0ab373af2` |
| URDF | `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96` |
| RobotSpec | `4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d` |
| Robot manifest | `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1` |
| TCP | `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18` |
| Foundations pixi.lock | `56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a` |
| Config | `f7064ba8ee1587aa90fbb04efc3c1899e381b2a19a9cf01d6b7b2e095e664d6d` |
| Samples JSONL | `085fdda74dbf89b9098f6a6a0627576eb2302fb705dd379288c9ff7a187e03dc` |

Gerçek argv/cwd/exit/UTC/stdout/stderr `commands/*/` ve
`clean-b/commands/*/` altındadır. Temiz tekrar başlangıcı
`2026-09-29T15:16:55.555071+00:00`, bitişi
`2026-09-29T15:20:24.637532+00:00` (driver duvar saati; performans ölçümü değil).
[COMMANDS](COMMANDS.md) kurulum, smoke, tam kabul ve salt okunur audit'i verir;
[reproduce_c103.py](../../../scripts/reproduce_c103.py) yeni checkout'ta
aynı 15 adımı fail-fast çalıştırır. Kullanıcı Word/PDF dosyaları kapsam dışıdır.

Stage2 metinleri canonical-LF hashlenir; gerçek log/JUnit baytları `-text` ile
korunur. [Manifest](evidence-manifest.json) raw çalışma ağacı SHA-256 ile
canonical-LF SHA-256'yı ayırır; [SHA256SUMS](SHA256SUMS) canonical değerleri
listeler. Git object SHA-1 aynı kimlik değildir. `full*/hashes.json` raw sayısal
kanıtı doğrular. Kapanış commit'i kendisini hashleyen belgelerin içine yazılmaz.

## Test ve ham kanıt

İki koşuda aşağıdaki bütün zorunlu kapılar PASS; eşikler değişmedi.

| Test | Ölçüm / sonuç | Dondurulmuş kabul |
|---|---|---|
| T-C01 float64, 1086 q | max konum L2 `4.47545209131181e-16 m`; max R Frobenius `8.763908156876301e-16` | Her biri ≤1e-9 |
| T-C01 float32, 1086 q | max konum L2 `1.7329206910447826e-7 m`; max R Frobenius `4.04344059421814e-7` | Her biri ≤1e-5 |
| T-C02 32 ayrı iç q, 2880 türev | max mutlak fark `5.289169102695723e-10`; 32 gradcheck PASS | h=1e-6; abs ≤1e-5 +1e-3 abs(FD) |
| Jacobian/frame | 32 geometrik/Pinocchio çapraz kontrol PASS | Frozen config |
| Sensitivity | 3 PASS; p ve R gradyana katkı veriyor | Frozen config |
| Batch | 1/2/7/32/1024 ×2 dtype; 2 batch graph PASS | Tek/batch eşliği |
| Edge | 30 edge-gradient PASS; 4 tekillik tanısı | Ayrı stencil; iç q'ya karıştırılmadı |
| Unit/negatif/arayüz | Her ortam 110 PASS, skip0; içindeki 21 C1-02 arayüz testi | Gerçek üretim API'leri |
| Kaynak mutasyonu | Her ortam 24 KILLED /0 SURVIVED /0 ERROR | Gerçek geçici kaynak diff + killing test |
| Foundations regresyon | Her ortam F0-01 16 +F0-02 102 +F0-03 159 =277 PASS | Eski testler değişmedi |
| Analitik ve smoke | 19 analitik; smoke 5 q ×2 dtype +2 gradient PASS | Tam ölçüm öncesi kapı |
| Temiz tekrar | 15 komut PASS, 11/11 exact artifact eş, raw sonuçlar aynı | Yeni checkout ve ortam |

En kötü gradient: `grad-0007`, component14 (smooth loss), joint index1;
autograd `0.5608984734188982`, FD `0.5608984728899813`.
Sıfıra yakın türevde en büyük göreli fark `1.0000148506627047`; birleşik
atol/rtol kapısının en büyük kullanım oranı `1.110223012299205e-5` (<1), dolayısıyla
geçerli PASS. Bu göreli fark saklanmadı; kabul yalnız göreli hataya dayanmaz.

Tam ham [birincil JSONL](full-exact-a/results.jsonl) ve
[temiz JSONL](clean-b/full/results.jsonl) her biri **2.317 satır /2.695.396 bayt**.
Bayt düzeyinde aynılar; SHA-256
`bacaf8e6cfccf9be7dec46fcffeffeeb392d619956746990ff547c267944e6d9`.
Her iki `failures.jsonl` boş (sıfır hata). Ham kanıt Git'te saklanır.
Örnek q, loss/bileşen, stencil, epsilon, autograd/FD ve hata dizileri JSONL'dedir.
Özet/JUnit/environment/hash dosyaları aynı dizinlerdedir;
[unit](unit-exact-a.xml), [clean unit](clean-b/unit.xml),
`mutations-exact-a/` ve `clean-b/mutations/` gerçek test/mutant kayıtlarıdır.
[acceptance.json](acceptance.json) iki koşuyu bağlayan makine okunur karardır.

## Sonuç ve yorum

**REQ-C02 / T-C01 / T-C02 PASS / ACCEPTED; C1-03 COMPLETE.**
Girdi → Torch kernel → bağımsız FK/FD/Jacobian testleri → iki ham koşu →
kabul zinciri tamamlandı. Birincil ve temiz koşular aynı kaynak/config/örnek
kimliğini taşır; [finalize_c103.py](../../../scripts/finalize_c103.py) kapanışı
JUnit, source, artifact, sample ve raw kanıttan denetledi (032/033 exit0).

İlk başarısız denemeler korundu: pip-check1 (Python 3.12 setuptools koşulu),
OpenMP abort3 ve analitik 18PASS/1FAIL (tarihsel eager import harness).
Kurulum subprocess exit0 sonrası konsol encoding hatası yalnız logger
yazdırmasındaydı. 005 erken analitik denemesi kabul değildir.
[ADR-012](../../../docs/adr/ADR-012-c103-runtime-overlay.md) ve
[protokol r2](PROTOCOL_R2.md) aynı NumPy sürümünün izole wheel'i, setuptools,
exact typing_extensions artifact'i ve harness düzeltmesini açıklar. Foundations
lock'u veya matematik eşikleri değiştirilmedi; OpenMP bypass kullanılmadı.
Önceki gelişim koşuları kabul koşularına birleştirilmedi. NaN mutasyonundaki
beklenen NumPy warning ve pytest JUnit property warning'leri loglarda durur.

Linux/CUDA/fiziksel robot **NOT_RUN**; performans ve etkin insan emeği
**NOT_MEASURED**; collision/safety **NOT_CHECKED**. Sonuç CPU Windows matematik
ve autograd sözleşmesi içindir. G1 kabulü veya v1.0.0 sürüm etiketi değildir.

## Sonraki adım

C1-04 için C1-02 kabulü ve bu doğrulanmış FK/config/örnek/kanıt girdileri hazır.
**C1-04 NOT_STARTED**; bu görevde neural eğitim veya sonraki faz başlatılmadı.
Stage1, Foundations ve C1-02 girdileri sonraki çalışmada da değişmez tutulur.
