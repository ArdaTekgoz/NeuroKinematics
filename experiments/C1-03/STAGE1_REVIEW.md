# C1-03 Aşama 1 — matematiksel sözleşme

29 Eylül 2026 · belge r1 · yazılım hedefi v1.0.0
Karar: **IN_PROGRESS / STAGE_1_COMPLETE**. REQ-C02; T-C01 ve T-C02 **NOT_RUN**.
Uygulama ve eğitim başlamadı. Açık Aşama 2 onayı gereklidir.

## Giriş ve incelenen kaynaklar

Başlangıç `main` HEAD ve `git ls-remote origin refs/heads/main` aynı:
`b82faf95ef87666471fd5ae77ec1d392364a108a`. Bu değer varsayılmadı, okundu.
İki kullanıcı Word dosyası izlenmiyor ve kapsam dışı. G0 24 Eylül PASS / ACCEPTED;
C1-02 29 Eylül COMPLETE / T-C07 PASS. Güncel kaynak C1-02 kapanışıdır.

AGENTS, görev kaydı, Core raporu §3–4, roadmap, TEST_PROTOCOL, TRACEABILITY,
STATUS, F0-01/02/03 çalışma ve yapı kayıtları, G0 kararı/devir manifesti,
C1-02 kabul raporu, mevcut chain/model/FK/Jacobian/finite-difference kodu,
ilgili testler ve Pixi/Python kilidi incelendi. `input-hashes.json` kimlikleri
saklar. `stage1-check.json` yalnız yapı/hash/örnekleme kontrolüdür.

## Robot ve frame sözleşmesi

KUKA KR 6 R900 sixx, `kuka_kr6_r900_sixx`; `joint_1`…`joint_6` sırası,
metre/radyan, sağ el, kolon vektörleri. Çıktı `T_base_link_tool0`;
`p=T[:3,3]`, `R=T[:3,:3]`. URDF origin `Trans(xyz) RotZ(yaw) RotY(pitch) RotX(roll)`;
her eklemde `T = T @ origin @ motion(q)`; axis origin sonrası joint-local eksendir.

| Eklem | Origin xyz (m) | Axis | Limitler (rad) |
|---|---|---|---|
| joint_1 | 0,0,0.400 | 0,0,-1 | -2.9670597283903604, 2.9670597283903604 |
| joint_2 | 0.025,0,0 | 0,1,0 | -3.3161255787892263, 0.7853981633974483 |
| joint_3 | 0.455,0,0 | 0,1,0 | -2.0943951023931953, 2.722713633111154 |
| joint_4 | 0,0,0.035 | -1,0,0 | -3.2288591161895095, 3.2288591161895095 |
| joint_5 | 0.420,0,0 | 0,1,0 | -2.0943951023931953, 2.0943951023931953 |
| joint_6 | 0.080,0,0 | -1,0,0 | -6.1086523819801535, 6.1086523819801535 |

Aktif origin rotasyonları sıfırdır. `joint_6-flange` fixed identity;
`flange-tool0` fixed `Ry(pi/2)` ve sıfır translation'dır: TCP **identity değildir**.
`base_link-base` yan dalı sonuç zincirine girmez. Dondurulmuş base dünya köküdür.
Sadece fixed/revolute desteklenir; continuous/prismatic/mimic/floating/planar
ve bilinmeyen tipler açık hata verir. Eksik limit/axis, bozuk ağaç/frame ve hash
uyuşmazlığı reddedilir. `load_robot` parse öncesi aynı baytları doğrular;
`extract_chain` fixed dönüşümleri korur. Parser ad kümesini kontrol eder;
bu nedenle yeni dış API ayrıca **tam sıralı tuple** ve kimlik doğrulamalıdır.

## Aday ve bağımlılık kararı

| Aday | Doğrulanan özellik | Maliyet/risk | Karar |
|---|---|---|---|
| `pytorch-kinematics==0.10.0` | MIT, Python >=3.9, Torch >=2.1; URDF serial root/end, fixed/revolute/continuous/prismatic; `.to(dtype,device)`; batch matrisi | Ek parser/frame dönüşümü ve geniş bağımlılık yüzeyi; tensor FK varsayılan analitik backward; bizim kesin kimlik/limit sözleşmemiz yine wrapper ister | İncelendi; kurulmadı, runtime NOT_RUN; seçilmedi |
| `native-torch-serial-v1` | Mevcut doğrulanmış parser, sınırlı joint kapsamı, standart Torch trigonometri/matmul | Yeni tensor hesabı ve gradyanı sıfırdan kabul edilmeli; NumPy kardeşi bağımsız oracle sayılamaz | **Seçildi**, Aşama 2 uygulaması bekler |

Aday wheel SHA-256 `62f78fecd60b29421bbab8854bcbb0704389b25058e33fc45d17b46e4b1b6788`
indirilen baytlardan doğrulandı. API: `build_serial_chain_from_urdf(data,end_link_name,root_link_name="")`,
`SerialChain.forward_kinematics(th,end_only=True).get_matrix()`; tensor yolu
`forward_kinematics_tensor(th,analytical_grad=True)`. Standart autograd için
`analytical_grad=False` seçeneği vardır. Bunlar 0.10.0 wheel kaynağından okundu,
yalnız hareketli master README'ye dayanılmadı. Metadata ve kaynak dosyası hashleri
[candidate-metadata.json](candidate-metadata.json) içinde. Kaynaklar:
[PyPI 0.10.0](https://pypi.org/project/pytorch-kinematics/0.10.0/),
[resmî depo](https://github.com/UM-ARM-Lab/pytorch_kinematics).

Seçilen runtime **torch==2.10.0+cpu**, CPython 3.12 Windows x64 wheel;
metadata lisansı BSD-3-Clause (`dependency-resolution.json`).
En yeni sürüm olduğu iddia edilmez; mevcut, exact ve CPU matematik kapsamına uygun
sürüm bilinçli seçildi. [Resmî CPU dağıtımı](https://download.pytorch.org/whl/cpu/torch/)
ve [kurulum tarifleri](https://pytorch.org/get-started/previous-versions/) doğrulandı.
`requirements-win-cpu.lock`, dokuz paketin URL/version/SHA kilididir;
`dependency-resolution.json` gerçek pip dry-run raporudur (host Python 3.11,
hedef `--python-version 3.12 --platform win_amd64`). Marker ortamı host olarak
raporlandığından kurulum uyumu henüz kanıt değildir; Aşama 2'de gerçek 3.12
ortamında `pip check` zorunludur. Wheel hashlerinin kaynağı `dependency-pins.json`da
ayrılır; Torch büyük wheel yalnız index metadata üzerinden pinlendi.

Foundations `pixi.lock` değişmez. Ayrı Windows venv, locked Pixi Python üzerinde
`--system-site-packages` ile yaratılacak; hashli Torch overlay oraya kurulacak.
Pinocchio 4.1.0 / NumPy 2.5.3 / Python 3.12.14 korunacak. Yeni Ortamın import,
DLL ve paket uyumu onay sonrası ön kapıdır. Mevcut ortam Torch içermez.
Linux/CUDA için destek veya hız iddiası yok. Mimari/ortam kararı
[ADR-011](../../docs/adr/ADR-011-c103-torch-fk.md) içindedir.

## Tasarlanan API ve autograd sınırı

Planlanan `TorchFK.from_frozen(root)` doğrulanmış robotu kurar; iç kernel
`TorchSerialChain` yalnız açık test fixture'larına da açılabilir. Sentetik robot
gerçek robot kimliği ile sunulamaz. Dış çağrı
`fk(q, *, robot_id, joint_names, units="rad")` tam kimlik/sıra/birim bekler.
Bir tensorun yanlış sıra veya derece içerdiği sayılardan her zaman anlaşılamaz;
metadata doğrulama bunu görünür yapar, sayısal mutasyonlar sessiz yanlış hesabı yakalar.

Yalnız dense, sonlu float64/float32 Torch tensor: `(6,) -> (4,4)`;
`(N,6) -> (N,4,4)`, N>=1. Liste, integer/complex/half, boş batch, yanlış rank,
NaN/Inf ve limit dışı q hata. Limitler giriş dtype'ında dışarı yuvarlanıp
genişletilmez; kontrol için q'nun float64 Torch görünümü kullanılır.
`q` kırpılmaz, derece dönüşümü yapılmaz. Dtype/device ve batch sırası korunur.

Sabit origin/axis/limit tamponları float64 ana kaynaktan input device/dtype'a
taşınır. Float32 tamponunu tekrar float64 yaparak hassasiyet kaybı gizlenmez.
Forward yalnız Torch `sin`, `cos`, `stack`, `cat`, broadcasting, matmul;
Rodrigues `R=I+sin(q)[a]x+(1-cos(q))[a]x²`. Yeni tensor üretiminde q'dan
`torch.tensor([...])` ile graph koparılmaz. In-place graph değişimi yok.
NumPy dönüşümü/detach/item/no_grad ve Pinocchio forward içinde yasaktır.
Hata kontrolünün boolean dallanması hesap grafiğinin yerine geçmez.
Sabitlerin constructor'da NumPy parser çıktısından kopyalanması q hesabı değildir.

Pinocchio mutable Data yalnız ayrı sayısal oracle süreç/instance'ındadır.
`T_world_base^-1 @ T_world_tcp` bağımsız referans olarak kalır. Torch ve NumPy
aynı parserı paylaşır; NumPy FK ve IndependentJacobian çapraz kontrol sağlar,
iki bağımsız doğruluk kanıtı olarak sayılmaz. Import yasağı testi Torch
forward'ın Pinocchio/NumPy FK'ye gizli bağımlılığını yakalayacaktır.

## Dondurulan deney ve riskler

[config](config.json), [örnekleme](SAMPLING_CONTRACT.md), [test matrisi](TEST_MATRIX.md)
ve [mutasyon matrisi](NEGATIVE_MUTATION_MATRIX.md) normatiftir. 1.086 q satırı,
32 bağımsız iç bölge gradient q'su; seçimde FK, Jacobian/SVD veya teacher sonucu
kullanılmadı. Wrist-singularity adayları yapısal seçildi; yakınlık niceliği henüz
**NOT_MEASURED**, Stage 2'de singular değerler raporlanır; aday silinmez.

C1-02 ham shardları bağımlılık değildir. 2.281 etiketsiz wide kayıt dışlama
nedeni değildir; burada teacher yerine limit içi q kullanılır. C1-02 dosyaları
değişmedi. Çıktı matris olduğu için quaternion branch türevi temel testte yoktur;
q/-q ve pi çevresi ayrı metrik regresyonudur. Genel bir konum gradyanı sıfır
olabilir (özellikle son eksen/TCP); her elemanı nonzero saymak yanlış kapıdır.
Tüm R bileşenleri, geometrik Jacobian ve kontrollü loss hassasiyeti birlikte aranır.

Hash/yapı kontrolü matematik doğruluğu değildir. Torch kurulum/runtime, smoke,
T-C01/T-C02, mutasyon/regresyon ve temiz tekrar üretim **NOT_RUN**. Maksimum
sayısal hatalar ve performans **NOT_MEASURED**. Onaydan önce C1-03 PASS yok.
