# F0-01 robot modeli ve manifest uygulama kaydı

Kimlik: RUN-20260918-002

Durum: PASS · TAMAMLANDI

Görev ve gereksinim: F0-01 · REQ-F01 · T-F01

Tarih ve sorumlu: 18 Eylül 2026 · Arda Tekgöz

## Soru ve değişiklik

Dondurulmuş KUKA KR 6 R900 sixx girdisi, geometri veya hash uydurmadan exact upstream commit'ten tekrar üretilebilir, Pinocchio tarafından parse edilebilir ve tam hash zincirli bir robot varlığına dönüştürülebilir mi?

Bu koşuda exact `kroshu/kuka_robot_descriptions` release `2.0.2`, commit `fbda927964caa1eb4e408fb0c25fe46b5a0bde3c` snapshot'ı minimum kapsamla dağıtıldı; Xacro 2.1.1 kilitlendi; çözülmüş URDF, canonical robot-spec ve manifest üretildi. T-F01 hem olumlu sözleşmeleri hem eksik limit/kaynak, değiştirilmiş dosya, hash uyuşmazlığı ve beklenmeyen varyant hata yollarını otomatik sınar.

Önceden sabitlenmiş kabul: standart suffixsiz varyant; Apache-2.0; `base_link`/`flange`/`tool0`; fixed-base açık seri zincir; `joint_1`–`joint_6`; kaynak eksen/limit izlenebilirliği; deterministik Xacro; tüm dosya ve mesh hashleri; Pinocchio 4.1.0 parse; canonical robot-spec hashinin yeniden üretimi. Bu eşikler sonuç görüldükten sonra düşürülmedi.

## Tekrar üretim

| Alan | Gerçek değer |
|---|---|
| Başlangıç HEAD | `ae9054c514c81b5ce237f89c86a9319b202c8745` |
| Başlangıç çalışma ağacı | `main...origin/main`; yalnız kullanıcıya ait izlenmeyen `~$uroKinematics_Model_Kullanim_Plani_r1.docx`; değiştirilmedi ve commitlenmedi |
| Uygulama commit'i | `4048c428afceaab4418d6107897dcd36c2d48f33` |
| Ortam | Native Windows 11 10.0.26200 x64; Pixi 0.81.0; CPython 3.12.14 |
| Araçlar | Xacro 2.1.1; PyYAML 6.0.3; Pinocchio 4.1.0; pytest 8.4.2 |
| Kaynak | `kroshu/kuka_robot_descriptions` 2.0.2 @ `fbda927964caa1eb4e408fb0c25fe46b5a0bde3c` |
| Kaynak lisansı | Apache-2.0; dağıtılan `source_snapshot/LICENSE` |
| Ortam kilidi | `pixi.toml` + `pixi.lock`; `win-64`, `linux-64`; yalnız Windows çalıştırıldı |
| Üretim komutu | `pixi run --locked build-robot-a` |
| Doğrulama komutu | `pixi run --locked verify-robot-a` |
| Test komutu | `pixi run --locked test-f01 --junitxml=experiments/F0-01/pytest-junit.xml` |
| Kabul koşusu | `2026-09-18T20:08:35.4924021+03:00` – `2026-09-18T20:08:39.0257895+03:00` |
| Tam emek süresi | Oturum başlangıcı zaman damgalanmadı; ölçülmedi |
| Seed/thread/GPU | Rastlantısal işlem yok; tek pytest süreci; GPU kullanılmadı |

Ana hashler:

| Varlık | SHA-256 |
|---|---|
| Çözülmüş URDF | `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96` |
| Canonical robot-spec | `4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d` |
| Manifest | `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1` |
| TCP sözleşmesi | `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18` |
| Kaynak sözleşmesi | `239979696395a118c6b0ac860588562868d18e6f9a7fc7fb293080715567f319` |

`robot_spec_sha256`, `robot_spec.json` dosyasının tamamının anahtarları leksikografik sıralı, ayraçları `,` ve `:`, UTF-8/LF ve tek final LF olan canonical JSON baytları üzerinden hesaplanır. Payload kendi hashini içermez; böylece döngü yoktur. Manifest hash'i dış kanıt olarak bu rapor ve `SHA256SUMS` içinde tutulur.

Değişiklik kapsamı: `.gitattributes`; `assets/robots/robot_a/`; `src/neurokinematics/robot_asset.py`; `tests/f0_01/`; Pixi manifest/lock; F0-01 ham kanıtları; `docs/SETUP.md`; zorunlu F0-00 kapanış uzlaştırması. Dosya ayrıntıları uygulama commit'inde izlenir.

## Test ve ham kanıt

| Kontrol | Ölçülen sonuç | Karar | Kanıt |
|---|---|---|---|
| Exact release/commit ve kritik kaynaklar | 7/7 raw-byte SHA-256 eşleşti | PASS | [`source-verification.json`](source-verification.json) |
| Xacro deterministik yeniden üretim | Art arda üretimler aynı URDF hashini verdi | PASS | [`COMMANDS.md`](COMMANDS.md), T-F01 |
| Manifest/hash/mesh | 29/29 dosya, 14/14 mesh URI | PASS | [`manifest-verification.json`](manifest-verification.json) |
| Robot yapısı | fixed-base, açık seri zincir, 6 revolute joint, beklenen sıra/eksen/limit | PASS | T-F01 |
| Pinocchio | 4.1.0 parse; `nq=6`, `nv=6` | PASS | `manifest-verification.json` |
| TCP ve canonical spec | Her iki hash yeniden üretildi | PASS | T-F01, [`SHA256SUMS`](SHA256SUMS) |
| Negatif hata yolları | Eksik limit/kaynak, değişiklik/hash uyuşmazlığı, yanlış varyant açık hata | PASS | [`pytest-junit.xml`](pytest-junit.xml) |
| T-F01 pytest | 16/16 PASS, 0.61 s | PASS | `pytest-junit.xml` |
| T-F00 regresyon | 6/6 PASS, 0.27 s | PASS | `COMMANDS.md` |
| Lock ve frozen install | `pixi lock --check`, `pixi install --locked`, exit 0 | PASS | `COMMANDS.md` |

İlk başarısız denemeler gizlenmedi:

- İlk normal Windows checkout CRLF dönüşümü nedeniyle kaynak hashlerini bozdu. Kabul edilmedi; snapshot `core.autocrlf=false` raw Git blob dışa aktarımıyla yeniden üretildi ve `.gitattributes` ile checkout baytları korundu.
- İlk resolved URDF, Xacro'nun rastgele geçici dizinini banner'a yazdığı için deterministik değildi. Rastgele banner çıkarılıp yalnız robot document element'i sabit XML bildirimiyle serileştirildi.
- Pinocchio dosya-yolu API'si checkout yolundaki `Ö` karakterini bozuk kodladı. Persisted UTF-8 URDF baytları Pinocchio'nun XML API'sine verilerek aynı model parse edildi.
- İlk toplu kapanış koşusunda kaynak raporu fonksiyonunun yanlış yerleşimi 15 PASS / 1 FAIL üretti. Kod sınırı düzeltildi ve frozen koşunun tamamı baştan sıfır hata ile tekrarlandı.

## Sonuç ve yorum

T-F01'in bütün zorunlu kontrolleri geçti. F0-01 kararı **PASS / KABUL / TAMAMLANDI**. Başarı belge varlığına değil, exact kaynak hashleri, deterministik üretim, manifest dosya doğrulaması, negatif testler ve Pinocchio parse sonucuna dayanır.

Açık sınırlamalar:

- KUKA üretici PDF'sinin raw baytları F0-00'da edinilemedi; belge hash'i `NOT_AVAILABLE` kalır. Exact upstream snapshot hash zinciri bundan etkilenmez.
- Linux yalnız dependency lock hedefidir; Linux kurulumu/testi `NOT_RUN`.
- Bu model-içi kinematik doğrulamadır; fiziksel doğruluk, kalibrasyon, collision, dinamik, payload ve robot güvenliği kanıtlanmadı.
- FK/Jacobian doğrulaması F0-01 kapsamında yapılmadı.

## Sonraki adım

F0-02'ye geçiş hazırdır; bu koşuda F0-02 başlatılmadı. Devredilecek immutable girdiler:

- `assets/robots/robot_a/robot.urdf` — `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96`;
- `assets/robots/robot_a/robot_spec.json` — `4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d`;
- `assets/robots/robot_a/manifest.json` — `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1`;
- `config/robots/tcp_tool0.json` — `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18`.

Kapanış commit'i bu raporu, görev/durum/roadmap ve izlenebilirlik kayıtlarını içeren ikinci committir; SHA'sı commit oluşturulduktan sonra Git ve remote kaydıyla raporlanır. Kendi commit SHA'sını aynı commit içine yazmak için üçüncü, yapay bir self-reference commit oluşturulmaz.
