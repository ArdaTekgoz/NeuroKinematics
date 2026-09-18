# NeuroKinematics Foundations kapsam ve ortam sözleşmesi

Durum: F0-00 için donduruldu

Yazılım hedefi: v0.1.0

Belge revizyonu: r1

Tarih: 18 Eylül 2026

Bu sözleşme REQ-F01 ve T-F00 için normatif girdidir. Sonuç görüldükten sonra tolerans, bütçe, frame, eklem sırası veya model kimliği değiştirilirse yeni belge revizyonu ve gerekirse ADR açılır; eski deney yeni sözleşmeyi geçmiş sayılmaz.

## 1. Kapsam

NeuroKinematics, robot başına ters kinematik adayları üreten ve sonuçları bağımsız kinematik hesaplarla doğrulayan bir araştırma ve mühendislik platformudur. Foundations kapsamı sabit tabanlı, açık seri zincirli, altı döner eklemli tek bir manipülatörün model-içi kinematik doğrulamasıdır.

İlk görev, `base_link` çerçevesinde `tool0` TCP konumu ve yönelimidir. Yalnız konum görevi ayrı bir profil tanımlanmadan bu sözleşmeye dahil değildir. Referans FK ve geometrik Jacobian Pinocchio ile, bağımsız kontrol sınırlı seri-zincir uygulamasıyla hesaplanacaktır.

## 2. Robot ve model girdisi

| Alan | Dondurulmuş değer |
|---|---|
| Robot | KUKA KR 6 R900 sixx, standart suffixsiz varyant |
| Hariç tutulan varyantlar | KR 6 R900-2; C, W ve WP varyantları |
| Model kimliği | `kuka_kr6_r900_sixx` |
| Kaynak | `kroshu/kuka_robot_descriptions` sürüm `2.0.2` |
| Kaynak commit | `fbda927964caa1eb4e408fb0c25fe46b5a0bde3c` |
| Kaynak lisansı | Apache-2.0; kök LICENSE ve `kuka_agilus_support/package.xml` aynı beyanı taşır |
| Üretici çapraz kontrolü | KUKA `0000-205-456`, V6.1, 23.09.2022; 6 eksen ve 901.5 mm azami erişim |
| Base | `base_link` |
| Kinematik tip | `flange` |
| Proje TCP'si | all-zero KUKA tool frame'i `tool0` |
| Etkin joint sırası | `joint_1` … `joint_6` |
| Karakteristik uzunluk ℓ | 0.9015 m; üretici azami erişimi, test verisinden öğrenilmez |
| Robot kaynak sözleşmesi SHA256 | `239979696395a118c6b0ac860588562868d18e6f9a7fc7fb293080715567f319` |
| TCP sözleşmesi SHA256 | `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18` |

Kaynak girdi ve kritik dosya hashleri [`config/robots/kuka_kr6_r900_sixx.source.toml`](../config/robots/kuka_kr6_r900_sixx.source.toml) içindedir. Bu model resmî bir KUKA URDF yayını değildir; sürümlü KUKA ROS 2 modelidir ve ana geometrisi resmî KUKA belgesiyle çapraz doğrulanmıştır. F0-01, exact snapshot'ı edinmeden, Xacro'yu çözmeden ve T-F01 yapısal denetimini geçmeden robot manifestini kabul edemez.

F0-00 hashleri kaynak ve sözleşme baytlarını kilitler. Tam snapshot dosya listesi, çözülmüş URDF hash'i ve canonical robot manifest hash'i F0-01 çıktısıdır; bunlar üretilmeden değer uydurulmaz.

## 3. Frame, dönüşüm ve TCP

- Sağ elli çerçeveler ve sütun vektörleri kullanılır.
- `T_A_B`, B çerçevesindeki koordinatları A çerçevesine taşır.
- Poz hedefi `base_link` içinde `tool0` pozudur.
- `T_base,TCP(q) = T_base,0 T_0,1(q1) … T_n-1,n(qn) T_flange,tool0`.
- `T_flange,tool0`: `xyz=[0,0,0] m`, `rpy=[0,+π/2,0] rad`, quaternion `wxyz=[0.7071067811865476,0,0.7071067811865475,0]`.
- TCP'nin makine-okunur tek kaynağı [`config/robots/tcp_tool0.json`](../config/robots/tcp_tool0.json) dosyasıdır.
- Tip link ile TCP örtük biçimde aynı kabul edilmez. Fiziksel takım eklenirse yeni TCP kimliği ve hash'i gerekir.

## 4. Birim, sayı ve poz temsili

- İç hesap birimleri metre, radyan ve saniyedir.
- Referans hesap türü IEEE-754 float64'tür.
- Dosya ve API sınırında quaternion sırası `w,x,y,z` olur.
- Quaternion girişleri tolerans içindeyse normalize edilir; sıfır norm, NaN ve Inf reddedilir.
- `q` ve `-q` aynı yönelimi temsil eder; işaret değişimi hareket değildir.
- Jacobian satır sırası önce çizgisel, sonra açısal hızdır. İki bölüm TCP noktasında ve base eksenlerinde ifade edilir.

## 5. Tolerans ve bütçe profilleri

| Profil | Konum | Yönelim | Kullanım |
|---|---:|---:|---|
| A | 2 mm | 1° | Ortak ana görev toleransı |
| B | 1 mm | 0.5° | Daha sıkı ayrı raporlama profili |

İki koşul birlikte sağlanmadan görev başarılı sayılmaz. Kinematik geçerlilik ayrıca sonlu/doğru boyutlu çıktı ve joint limit uyumu gerektirir.

Zaman bütçeleri 10 ms ve 50 ms olarak ayrı profillerdir. DLS için iterasyon üst sınırı 200'dür. Timeout ile geometrik başarı ayrı kaydedilir. Bunlar T-F00 performans hedefi değildir; T-F00 yalnız sözleşme ve ortamın çalıştığını denetler.

## 6. Çalışma ortamı

| Alan | Karar |
|---|---|
| Kanonik F0 geliştirme/test hostu | Native Windows 11 x64 |
| Bu koşunun exact hostu | Windows 11 Pro 10.0.26200, x64 |
| Ortam yöneticisi | Pixi 0.81.0 |
| Paket kanalı | conda-forge |
| Python | CPython 3.12 serisi; exact patch `pixi.lock` içinde |
| Referans backend | Pinocchio 4.1.0 |
| Test aracı | pytest; exact sürüm `pixi.lock` içinde |
| Desteklenen lock platformları | `win-64`, `linux-64` |
| Çalıştırılmış platform | Yalnız `win-64`; Linux sonucu iddia edilmez |
| Hesap | CPU doğrulaması; GPU T-F00 şartı değildir |

WSL bu hostta kurulu değildir ve F0 sözleşmesinin kanonik ortamı değildir. Linux bağımlılık çözümü kilitte tutulur; gerçek Linux desteği ancak ayrı bir çalıştırılmış kanıtla verilir. Kurulum ve frozen çalışma tarifi [`docs/SETUP.md`](SETUP.md) içindedir. Platform kararının gerekçesi [ADR-004](adr/ADR-004-platform-ve-varsayimlar.md) ile kaydedilir.

## 7. Açık hata kuralları

Aşağıdaki durumlarda gizli varsayılanla devam edilmez:

- model kaynağı, lisansı, exact commit'i veya beklenen hash'i uyuşmaz;
- base, tip, TCP veya joint sırası eksiktir;
- desteklenmeyen joint türü ya da eksik limit vardır;
- birim/frame/quaternion sözleşmesi belirsizdir;
- model/TCP hash'i beklenenden farklıdır;
- Python, Pinocchio veya mimari ortam sözleşmesini karşılamaz.

Sayısal çözücünün başarısızlığı hedefin erişilemez olduğunu tek başına kanıtlamaz.

## 8. Kapsam dışı ve iddia sınırları

- floating base, kapalı çevrim, mimic joint, çoklu kol ve genel amaçlı robotik kütüphane;
- neural model eğitimi, Torch FK, ONNX, GUI ve ikinci robot;
- çarpışma, dinamik uygulanabilirlik, payload, eyleyici/effort ve inertia doğrulaması;
- fiziksel kalibrasyon, gerçek robot kontrolü ve safety-certified davranış;
- model-içi doğrulamayı fiziksel doğruluk veya çarpışmasızlık kanıtı olarak sunmak;
- GPU performansı, deadline garantisi veya Linux çalışabilirliği iddiası.

## 9. F0-00 kabul bağı

T-F00 şu koşulların tümünde PASS olabilir:

1. Bu sözleşmenin zorunlu alanları ve makine-okunur robot/TCP girdileri doludur.
2. `pixi.lock` iki hedef platform için günceldir ve `pixi install --locked` Windows ortamını kurar.
3. Ortam kontrolü Python 3.12, 64-bit süreç, float64, NumPy ve Pinocchio 4.1.0 smoke kontrolünü geçirir.
4. F0-00 pytest kümesi sıfır hata ile biter.
5. Gerçek komut, ortam envanteri, hashler ve ham log `experiments/F0-00/` altında saklanır.

F0-00'ın kabulü F0-01'i otomatik kabul etmez. F0-01 exact robot snapshot'ı, çözülmüş URDF'yi, tam manifesti ve T-F01 yapısal doğrulamasını üretir.
