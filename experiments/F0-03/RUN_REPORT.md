# F0-03 Jacobian ve metrik doğrulama kaydı

Kimlik: RUN-20260919-001
Durum: PASS · KABUL · TAMAMLANDI
Görev ve gereksinim: F0-03 · REQ-F03 · T-F03 / T-F04
Tarih ve sorumlu: 19 Eylül 2026 · Codex; proje sahibi Arda Tekgöz
Yazılım hedefi: v0.1.0 · Belge revizyonu: r2

## Soru ve değişiklik

Aynı immutable URDF için bağımsız geometrik, Pinocchio ve merkezi sonlu fark
Jacobianları TCP noktasında, base eksenlerinde, `[linear; angular]` düzeninde
uyuşuyor mu? Poz, SO(3), wxyz quaternion ve normalize SVD metrikleri analitik
beklentilerle doğrulandı. Dondurulmuş ana kabul maksimum normalize fark ≤1e-5;
256 deterministik q, seed 20260919, h=1e-6 rad; h=1e-5 ve 1e-7 duyarlılık
koşuları. Eşik, seed, h, örnek sayısı veya örnekleme sonuç görüldükten sonra
değiştirilmedi. Ek 21 elle seçilmiş q de aynı üç h için kabul kapısına dahildir.

İlk denemede görev metnindeki 62 karakterlik hatalı hash nedeniyle güvenli
durdurma uygulandı; F0-02 kanıtı ve dosyanın doğrudan SHA-256 ölçümüyle doğru
64 karakterlik değer teyit edilerek çalışma sürdürüldü. Kullanıcı doğru TCP
hash'ini açıkça yetkilendirdi. Bu olay kaynak dosya değişikliği, bütünlük ihlali
veya F0-02 regresyonu değildir. `tcp_tool0.json` yeniden yazılmadı. Tarihsel
[`preflight.json`](preflight.json) ilk durdurmayı, [`authorized-preflight.json`](authorized-preflight.json)
yetkilendirilmiş devam kontrolünü korur. Ayrı blocked commit oluşturulmadı.

F0-02 API ve modülleri değişmedi. `jacobian.py` yalnız bağımsız FK/zincir
verilerinden her origin/axis'i base'e taşır ve `z × (p_tcp-p_i)` kullanır.
Fixed eklemler dönüşümde kalır, sütun üretmez. `finite_difference.py` yalnız
IndependentFK çağırır. Ayrı süreç testi Pinocchio importunu yasaklayarak bu iki
yolu çalıştırır. Referans modeli/placement/Jacobian bağımsız yollara girmez.

`pinocchio_jacobian.py` F0-02 adaptörünün immutable XML, BODY frame ve idx_q
eşlemesini kullanır; sütunlar ayrıca idx_v ile eşlenir. Pinocchio 4.1.0
`LOCAL_WORLD_ALIGNED` TCP noktasında dünya eksenleri verir. Her sütun `Motion`
olarak yorumlanır, `.linear` ve `.angular` ayrı alınır; iki bileşen
`R_world_base.T` ile base eksenlerine döndürülür. Ham satır sırası varsayılmaz.
WORLD farklı noktayı, LOCAL farklı eksenleri kullanır. Bu ayrım
[resmî referans-frame tanımları](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/group__pinocchio__multibody.html),
kurulu 4.1.0 docstring'i ve nonidentity base/TCP analitik testleriyle doğrulandı.

Merkezi fark açısal bölümü `log(R_plus @ R_minus.T)/(2h)` ile base eksenlerindedir.
Vee yönü `(R32-R23,R13-R31,R21-R12)/2`; küçük açıda seri açılım, pi yakınında
simetrik matristen birim özvektör ve skew işareti kullanılır. Tam pi'de eksen
işareti belirsizdir; en büyük bileşeni pozitif seçilir. Euler türevi kullanılmaz.
İç marj `2e-5 rad`; hiçbir q±h kırpılmaz veya yeniden örneklenmez.

## Tekrar üretim

| Alan | Değer |
|---|---|
| Başlangıç HEAD | `34947308d31ccaf55bd8f74640bc12d6834dc605` |
| Uygulama commit'i | 981f6143ce38574021edac7373586976cf97bdf4 |
| Başlangıç Git | main; fetch başarılı; ahead/behind 0/0; yalnız önceki geçici ENGELLİ kayıtları ve kullanıcı Word lock dosyası |
| Remote ön koşul | F0-02 uygulama ve kapanış commit'leri origin/main ataları; runner yeniden kontrol eder |
| Ortam | Windows 11 / Python 3.12.14 / NumPy 2.5.3 / Pinocchio 4.1.0 / Pixi 0.81.0 |
| Paket kilidi | Yeni bağımlılık yok; pixi.lock değişmedi; SHA-256 `56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a` |
| CPU/thread/RAM/GPU | environment.json; seri CPU döngüsü; BLAS gerçek thread sayısı ve RAM ölçülmedi; GPU kullanılmadı |
| Frames/sıra/birim | base_link / flange / tool0; joint_1…joint_6; metre/radyan/saniye; float64 |
| Ölçek | ℓ=0.9015 m; yalnız doğrulanmış manifestten okunur; config eşitliği kontrol edilir |
| Config | config.json; SHA-256 `4453a8e8ad6c4930505af8ce10a080292e9db4a1bab5a71e50af9d44dce2c9ce` |
| Örnekleme | NumPy Generator / PCG64; seed 20260919; uniform(lower+2e-5,upper-2e-5), (256,6) |
| Örnek hash | `678eb4286863026880792ef0cc3c0a9d4f92e16f85b1aa009705cbf0b59b26e7` |
| Hash formatı | C-order little-endian float64 raw baytlar; metadata hash'e girmez |
| F0-02 devralınan sample | `8fb7e88758aa841310ae4d665d76d00a4488a5b79217ca4d5c80a825715c7101`; runner yeniden üretti |
| Kapanış komutu | `pixi run --locked python scripts/run_f03_acceptance.py` |
| Süre | UTC başlangıç/bitişler commands.json; toplam etkin emek ölçülmedi; benchmark iddiası yok |
| Dataset/split/model/checkpoint | Uygulanamaz; F0-04/IK/ML başlatılmadı |

| Immutable girdi | SHA-256 |
|---|---|
| `assets/robots/robot_a/manifest.json` | `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1` |
| `assets/robots/robot_a/robot.urdf` | `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96` |
| `assets/robots/robot_a/robot_spec.json` | `4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d` |
| `config/robots/tcp_tool0.json` | `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18` |

## Test ve ham kanıt

| Kontrol | Sonuç | Kanıt |
|---|---|---|
| Locked ortam ve 10 kapanış komutu | Tümü exit 0 | commands.json |
| F0-00 | 6/6 PASS | f00-junit.xml |
| F0-01 | 16/16 PASS; varlık kontrolü PASS | f01-junit.xml, robot-verification.json |
| F0-02 | 102/102 PASS; 10000 q dahil | f02-junit.xml |
| F0-03 unit | 156/156 PASS | unit-junit.xml |
| T-F03 tam kabul | Her üç h için 256+21 q; üç çift PASS | jacobian-validation-summary.json, jacobian-sensitivity.json |
| T-F04 | 48/48 PASS | metrics-junit.xml, metric-validation-summary.json |
| F0-03 tam pytest | 159/159 PASS; üç kabul testi normal komutta çalışır | pytest-junit.xml |
| Mutasyon | 12/12 gerçekten yakalandı | mutation-results.json, JUnit properties |
| Kanıt bütünlüğü | SHA256SUMS dosyaları raw bayt üzerinden doğrulandı | runner --verify-only |

256 deterministik örnek için maksimum normalize farklar:

| h (rad) | Geometrik–Pinocchio | Geometrik–merkezi | Pinocchio–merkezi |
|---|---:|---:|---:|
| 1e-05 | 2.9189048438105259e-16 | 2.0189103292622298e-11 | 2.0189176475996084e-11 |
| 1e-06 | 2.9189048438105259e-16 | 1.7676058530094515e-10 | 1.7676048885535101e-10 |
| 1e-07 | 2.9189048438105259e-16 | 2.1389089436208680e-09 | 2.1389089536102113e-09 |

Her çift/h için N, geçerli/geçersiz/nonfinite sayıları, min/median/P95/P99/max,
en kötü index/q ve üç yöntemin tekillik özetleri JSON içindedir. Percentile
yöntemi NumPy `linear`. Aşım, geçersiz, nonfinite ve eşiğin %90'ına yaklaşan
örnek sayıları sıfır. Başarısız örnek silme/yenileme veya alt küme seçimi yok.

Ana h=1e-6 koşusunda en kötü fark, sıfır tabanlı **247** numaralı q'dadır:
`[-2.3988157277773476, -2.464675351444274, -1.5234317337241583,
2.976288858472781, -0.31625485371385764, 2.252794626580921] rad`.
Geometrik–Pinocchio en kötü index **41**. Duyarlılıkta merkezi fark çiftlerinin
en kötü indexleri h=1e-5 için **165**, h=1e-7 için **112**; tam q'lar JSON'da.

Ek 21 kabul konfigürasyonu; sıfır, limit orta noktası, üç karışık q, 12 tek
eklem ±0.25 rad, iki limit iç yanı, wrist hizalı ve wrist hizasına yakın q'dur.
Tüm matrisleri `handpicked-jacobians.json` içinde bulunur. h=1e-7'de bu grupta
en büyük fark **2.327181019633208e-9**, `lower_interior` örneğindedir:
`[-2.9670397283903602, -3.316105578789226, -2.094375102393195,
-3.2288391161895094, -2.094375102393195, -6.108632381980153] rad`.
Bu ek örnek de kabulden çıkarılmadı; ana h'de en kötü örnek yine index 247'dir.

256 örnek, bağımsız geometrik normalize Jacobian tekillik dağılımları:

| Metrik | Min | Median | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| condition | 5.69389459159 | 14.0226960145 | 235.176972825 | 1153.4503185 | 98964.6468259 |
| manipulability | 6.45460998149e-06 | 0.0498033014026 | 0.143970665369 | 0.170005242511 | 0.181721358287 |
| sigma_max | 1.6250997579 | 1.85689275948 | 1.96675127422 | 1.99057939398 | 2.00549486867 |
| sigma_min | 1.77036820925e-05 | 0.133089086117 | 0.272406157872 | 0.304476950143 | 0.320293769183 |

Sıfır q'da geometrik ve Pinocchio sigma_min=0, condition=Inf,
manipulability=0. JSON Inf'i açık `"Inf"` metniyle taşır. Küçük pozitif
singular value gizli epsilonla büyütülmez veya sıfıra kırpılmaz; condition oran,
manipulability doğrudan SVD çarpımıdır. Sayısal rank (`6*eps*sigma_max`) yalnız
teşhistir ve metrikleri değiştirmez. Wrist hizasında geometrik sigma_min
4.2465670677817183e-17 iken merkezi fark 7.160669176523449e-12 verir (h=1e-6):
tekillikte condition/metrik duyarlılığı açıkça `diagnostics.json` alt grubunda
korunur. Jacobian farkları yine kabul sınırının altındadır. Analitik sıfır,
rank-deficient, diagonal ve 1e-12/1e-18 near-singular matrisler ayrı sınandı.

T-F04: 0/90/180° x/y/z dönüşleri, trace clip, wxyz sırası, q/-q eşdeğerliği,
norm toleransı/copy davranışı, sıfır/NaN/Inf/şekil reddi, determinant +1 ve
ortonormallik kontrolü geçti. Konum 0.003/0.004 m örneğinde 0.005 m = 5 mm;
açı sunumunda rad→derece açık. SO(3) log için sıfır, küçük açı, pi yakını ve
sol artışın base eksenlerinde olması ayrıca sınandı.

| Gereksinim | Değişiklik | Test / kanıt |
|---|---|---|
| Bağımsız joint origin/axis, cross, fixed ve TCP | jacobian.py | test_analytic.py, test_mutations.py |
| Pinocchio frame ve sütun/satır sözleşmesi | pinocchio_jacobian.py | nonidentity base, idx_q/idx_v, LOCAL/WORLD/LWA testleri |
| SO(3) merkezi fark ve limit | finite_difference.py | küçük/pi açı, 2. derece yakınsama, limit reddi |
| Metrikler ve ölçek | metrics.py | test_metrics.py; T-F04 JSON/JUnit |
| Determinizm ve hata koruma | jacobian_validation.py | test_evidence.py; fake nonfinite/yanlış/eşik yakını sonuçlar |
| Gerçek robot kapısı | test_acceptance.py | 256+21 q × 3 h; test-f03 içinde atlanmaz |
| Regresyon ve kanıt bütünlüğü | run_f03_acceptance.py | F0-00/01/02 JUnit; bozulmuş hash reddi testi |

12 mutasyon: satır tersliği, ters cross, flange noktası, local axis, world/base,
ters joint sırası, fixed atlama, axis işareti, Pinocchio LOCAL, Euler türevi,
forward difference, derece/radyan. Üretim dosyaları değiştirilmedi. Forward
difference h=1e-6'da tek başına 1e-5 robot sınırından kaçabilir; daha sıkı
analitik merkezi-fark doğruluk testinde 2.5572910745379003e-6 > 1e-8 ile yakalandı.
Ayrıca h yarıya inince merkezi fark hatasının yaklaşık dört kat küçülmesi test edildi.

Geliştirme denemeleri: İlk unit koşusu 148 PASS / 6 FAIL verdi; üçü EigenPy'nin
tek sütunu squeeze etmesi, üçü 90° quaternion dönüşümünde 3.33066907e-16
yuvarlama farkıydı. Referans matris reshape ve simetrik squared-component
quaternion formülüyle düzeltildi; 3e-16 dahil test sınırları gevşetilmedi.
`development-unit-junit.xml` başarısız ilk koşuyu, `development-fixed-unit-junit.xml`
154/154 düzeltme koşusunu korur. Son incelemede manipulability'nin rank
teşhisiyle kırpılması kaldırıldı; 1e-18 ve bozuk kanıt testi eklendi. Nihai
156 unit / 159 tam test ve 48 metrik testi yeniden geçti.

## Sonuç ve yorum

**F0-03 PASS / KABUL / TAMAMLANDI.** T-F03 ve T-F04 birlikte, immutable girdiler,
regresyonlar ve mutasyonlar korunarak geçti. Sayısal hatalar fiziksel robot
hassasiyeti değildir. Kaynak raporlar, F0-02 kanıtları, robot dosyaları ve
kullanıcıya ait Word lock dosyası korunmuştur.

Açık sınırlar: yalnız Windows çalıştırıldı; Linux NOT_RUN. Ortak yanlış fiziksel
URDF bu karşılaştırmayla dışlanamaz; kalibrasyon/collision/dinamik/robot güvenliği
doğrulanmadı. Pinocchio Data instance'ı eşzamanlı paylaşılmaz. Pi log eksen
işareti matematiksel olarak belirsizdir. Normu 1'den 1e-6'dan fazla sapan
quaternion reddedilir. Near-singular condition sonlu fark yuvarlamasına duyarlıdır;
bu örnekler saklanmıştır. Üretici PDF bayt/hash eksikliği F0-01'den devralınır.
Benchmark/deadline garantisi verilmez. SHA256SUMS kendisini ve kapanış Markdown
anlatısını dışlar; kaynak/test/runner/pixi ve JSON/JUnit raw kanıtlarını kapsar.
Sample/config/sayısal sonuçlar yeniden üretilebilir; JUnit zamanları ve HEAD
kaydı koşuya bağlıdır. Raw 256-q matrisi config/seed ile üretilir, ayrı dataset değildir.

## Sonraki adım

**F0-04'e geçiş hazır; F0-04 başlatılmadı.** Foundations G0 kapanmadı.
Uygulama ve ham kanıtlar ilk commit'te; bu rapor ve durum/izlenebilirlik/roadmap
ikinci commit'tedir. Kapanış commit'inin kendi SHA'sı üçüncü commit gerektirmemek
için dosyaya yazılmaz; iki SHA ve gerçek push sonucu kullanıcı yanıtında verilir.
