# ADR 005 KR 6 R900 sixx model girdisi ve görev sınırı

Durum: Kabul edildi

Karar tarihi: 18 Eylül 2026

Etkilenen görevler: F0-00, F0-01

## Bağlam

İlk robot daha önce yalnız “KUKA KR6 R900 sixx” planlama varsayımıydı. Exact varyant, model kaynağı, lisans, base/tip/TCP, joint sırası, karakteristik uzunluk ve hash yöntemi belirlenmeden tekrar üretilebilir F0-01 çalışması başlayamaz.

Roadmap robot manifesti ve nihai model hashlerini F0-01'e verir. F0-00'ın açık engelleri kapatabilmesi için F0-01'in girdilerinin daha önce dondurulması gerekir. Bu karar doğrulamayı öne çekmez; yalnız kaynak edinme sözleşmesini öne çeker.

## Karar

- Exact varyant suffixsiz standart **KUKA KR 6 R900 sixx**'tir; R900-2, C, W ve WP değildir.
- Model girdisi `kroshu/kuka_robot_descriptions` sürüm `2.0.2`, commit `fbda927964caa1eb4e408fb0c25fe46b5a0bde3c`, paket `kuka_agilus_support` olarak sabitlenir.
- Kaynak Apache-2.0'dır. Model resmî KUKA URDF yayını olarak adlandırılmaz; sürümlü KUKA ROS 2 kaynağıdır.
- Kinematik ölçüler ve eklem limitleri KUKA'nın exact ürün belgesiyle çapraz doğrulanır.
- Base `base_link`, kinematik tip `flange`, proje TCP'si `tool0` olur. Sabit dönüşüm `xyz=[0,0,0]`, `rpy=[0,+π/2,0]`'dır.
- Etkin joint sırası `joint_1` ile `joint_6` arasındadır.
- Karakteristik uzunluk üreticinin 901.5 mm azami erişiminden `0.9015 m` alınır; deney verisinden öğrenilmez.
- SHA-256 ve raw-byte hashing kullanılır. Metin sözleşmeleri UTF-8/LF tutulur.
- F0-00 kritik upstream dosya hashlerini, TCP sözleşmesini ve kaynak sözleşmesini kilitler.
- F0-01 exact snapshot'ı edinir, tüm dağıtılan varlıkları hashler, Xacro'yu kilitli araçla çözer ve `resolved_urdf_sha256` ile canonical `robot_spec_sha256` üretir. Bu değerler F0-00'da uydurulmaz.

## Görev sınırına etkisi

F0-00'a makine-okunur kaynak ve TCP girdi sözleşmesi eklenmiştir. F0-01'in `assets/robots/robot_a/manifest.json`, model doğrulaması ve T-F01 kabul sorumluluğu değişmemiştir. Böylece gereksinim → değişiklik → test → kanıt sırası korunurken F0-01 belirsiz bir kaynaktan başlamaz.

## Reddedilen alternatifler

- Git geçmişindeki kaynaksız “simplified” KR6 URDF: ölçüleri, joint adları, eksenleri ve TCP'si seçilen kaynakla çelişir; lisans/provenans taşımaz.
- `ros-industrial/kuka_experimental`: model yararlı bir tarihsel kaynaktır fakat depo kökü ile paket metadata lisans beyanları aynı değildir ve paket deneysel olarak işaretlidir.
- KR 6 R900-2: farklı ürün varyantıdır; sessiz ikame edilemez.
- Bellekten geometri veya DH tablosu üretmek: güvenilir kaynak ve hash zincirini bozar.

## Sonuçlar ve engel kuralı

Kaynak edinme engeli F0-01'i başlatmak için koşullu olarak çözülmüştür. F0-01 sırasında herhangi bir kaynak hash'i uyuşmaz, Xacro çözülemez veya yapısal/üretici çapraz kontrolü başarısız olursa F0-01 `ENGELLİ`/`FAIL` kalır. Robot değişikliği ancak yeni ADR ile yapılır.
