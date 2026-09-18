# F0-00 kapsam ve ortam uygulama kaydı

Kimlik: RUN-20260918-001

Durum: PASS · TAMAMLANDI

Görev ve gereksinim: F0-00 · REQ-F01 · T-F00

Tarih ve sorumlu: 18 Eylül 2026 · Arda Tekgöz

## Soru ve değişiklik

F0-00'ın kapsam, model girdisi ve çalışma ortamı belirsizlikleri tekrar üretilebilir biçimde kapatılabilir mi; kilitli küçük Python ortamı native Windows hostunda gerçekten çalışır mı?

Önceki durumda yalnız plan belgeleri vardı. Bu koşuda:

- kapsam, birim, frame/TCP, quaternion, tolerans, bütçe ve iddia sınırları `docs/SPEC.md` ile donduruldu;
- platform kararı ADR-004, robot kaynak/girdi kararı ADR-005 ile kaydedildi;
- makine-okunur robot kaynak ve TCP sözleşmeleri oluşturuldu;
- Python `src/` proje yapısı, Pixi manifesti ve iki platformlu lock üretildi;
- JSON ortam kontrolü ile altı pytest kabul kontrolü uygulandı.

Önceden sabitlenmiş kabul: SPEC zorunlu alanları dolu olacak; lock `win-64` ve `linux-64` için çözülecek; native Windows ortamı lock değiştirilmeden kurulacak; Python 3.12, 64-bit, NumPy float64, Pinocchio 4.1.0 ve `SE3.Identity()` kontrolleri ile F0-00 test kümesi sıfır hata verecek.

## Tekrar üretim

| Alan | Gerçek değer |
|---|---|
| HEAD | `9b51aefcc6c1e87a7be36c8c0b055c91144ee86d` |
| Çalışma ağacı | Bu koşunun değişiklikleri commitlenmedi; başlangıçtaki kullanıcıya ait `~$uroKinematics_Model_Kullanim_Plani_r1.docx` kapsam dışı bırakıldı |
| Ortam yöneticisi | Pixi 0.81.0 |
| Ortam kilidi | `pixi.lock`; exact hash [`SHA256SUMS`](SHA256SUMS) içinde |
| İşletim sistemi | Microsoft Windows 11 Pro 10.0.26200 build 26200, x64 |
| CPU | AMD Ryzen 7 250 w/ Radeon 780M Graphics, 8 fiziksel / 16 mantıksal çekirdek |
| RAM | 25,025,695,744 bayt |
| GPU | NVIDIA RTX 5060 Laptop GPU 8151 MiB + AMD Radeon 780M; kullanılmadı, performans ölçülmedi |
| Python | CPython 3.12.14, 64-bit |
| Temel paketler | Pinocchio 4.1.0; NumPy 2.5.3; pytest 8.4.2; Hatchling 1.27.0 |
| Thread/runtime | T-F00 thread sayısı sabitlenmedi; tek Python test süreci, CPU smoke kontrolü |
| Robot kaynak sözleşmesi | `239979696395a118c6b0ac860588562868d18e6f9a7fc7fb293080715567f319` |
| TCP sözleşmesi | `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18` |
| Veri/split/model/checkpoint | Üretilmedi; F0-00 için uygulanamaz |
| Config | `docs/SPEC.md`, `config/robots/*.toml|json`, `pixi.toml`, `pixi.lock` |
| Seed | Rastlantısal işlem yok; uygulanamaz |
| Tam süre | Oturum başlangıcı zaman damgalanmadı; araç kurulumu başlangıcı ile son kabul arasında gözlenen alt sınır 33 dk 16 sn |
| Kabul koşusu | 11:48:34.4071413+03:00 – 11:48:41.8889434+03:00 |

Kurulum ve tekrar komutları [`docs/SETUP.md`](../../docs/SETUP.md), gerçek komut/çıkış özeti [`COMMANDS.md`](COMMANDS.md), donanım ayrıntısı [`system-inventory.json`](system-inventory.json) içindedir.

## Test ve ham kanıt

### T-F00 ortam kontrolü

Komut:

```powershell
pixi run --locked env-check --output experiments/F0-00/environment.json
```

Ham çıktı: [`environment.json`](environment.json). Çıkış kodu `0`, durum `PASS`.

Geçen kontroller:

- CPython `3.12.14`, beklenen `3.12.*`;
- 64-bit süreç;
- NumPy `2.5.3` importu ve 64-bit `float64`;
- Pinocchio `4.1.0` importu;
- `pinocchio.SE3.Identity()` için exact 4×4 identity smoke kontrolü.

### Robot kaynak bayt kontrolü

Exact `2.0.2` release commit'indeki yedi kritik Xacro/YAML/lisans girdisi ağ üzerinden yeniden alındı ve sözleşmedeki SHA-256 değerlerinin tamamıyla eşleşti. Ham sonuç: [`source-verification.json`](source-verification.json), durum `PASS`.

KUKA datasheet'inin resmî URL'si doğrudan istemcide PDF yerine dinamik HTML indirme portalına yönlendi. Değişken HTML hash'i PDF hash'i olarak kabul edilmedi; üretici belge kimliği ve resmî indeks içeriği kaynak kaydında korunurken document SHA256 `NOT_AVAILABLE` bırakıldı. Bu sınırlama nihai model dosyalarının exact upstream commit ve hash zincirini etkilemez.

### T-F00 pytest kabul kümesi

İlk çağrı `pytest` console-script sarmalayıcısıyla yapıldı ve kullanıcı yolundaki `Ö` karakterinin bozuk kodlanması nedeniyle test gövdesine ulaşmadan `101` ile çıktı. Uygulama görevi kalıcı olarak `python -m pytest` biçimine çevrildi; bu başarısız deneme [`COMMANDS.md`](COMMANDS.md) içinde saklandı.

Nihai komut:

```powershell
pixi lock --check
pixi install --locked
pixi run --locked test-f00 --junitxml=experiments/F0-00/pytest-junit.xml
```

Sonuç: lock güncel, locked install, ortam kontrolü ve kaynak doğrulaması başarılı, `6 passed in 0.28s`; beş komutun çıkış kodu `0`. Ham pytest/JUnit kanıtı: [`pytest-junit.xml`](pytest-junit.xml).

Altı test; ortam sözleşmesini, JSON CLI çıktısını ve hata davranışını, exact robot kaynak kararını, TCP dönüşüm/hashini ve SPEC bölüm bütünlüğünü kapsar.

### Kabul değerlendirmesi

| Ölçüt | Ölçülen sonuç | Karar |
|---|---|---|
| Zorunlu SPEC alanları | Testle mevcut | PASS |
| `win-64` + `linux-64` lock çözümü | `pixi lock` başarılı | PASS |
| Lock değiştirmeden Windows kurulum | `pixi install --locked`, exit 0 | PASS |
| Küçük Python/Pinocchio kontrolü | Tüm alt kontroller true | PASS |
| F0-00 test kümesi | 6/6 PASS | PASS |

## Sonuç ve yorum

F0-00 kabul ölçütleri karşılandı. Native Windows 11 x64 ortamı gerçek hostta kurulmuş, bağımlılık kilidi iki hedef platform için çözülmüş ve T-F00 çalıştırılmıştır. Başarılı sonuç belge varlığından değil, JSON ve JUnit ham kanıtından alınmıştır.

Robot kaynağı engeli F0-01'i başlatmaya yetecek kadar kapatıldı: exact standart KR 6 R900 sixx varyantı, sürüm/commit/lisans, frame/TCP, joint sırası, ℓ ve kritik upstream hashleri sabittir. Bununla birlikte şu sonuçlar **üretilmedi** ve iddia edilmez:

- Linux üzerinde kurulum veya test (`linux-64` yalnız çözüldü);
- complete upstream snapshot ve mesh manifesti;
- çözülmüş URDF ve onun hash'i;
- canonical nihai robot manifest hash'i;
- T-F01 model doğrulaması;
- FK, Jacobian, performans, GPU, collision, fiziksel doğruluk veya robot güvenliği.

İlk install sırasında cache açılımı bir kez I/O uyarısı verdi ve otomatik retry ile tamamlandı. İkinci locked install temiz biçimde başarılı oldu; tekrar oluşursa Pixi cache yolu ayrıca teşhis edilmelidir.

## Sonraki adım

F0-01 açılabilir. Korunacak girdiler:

- `docs/SPEC.md` ve ADR-005 kararları;
- `config/robots/kuka_kr6_r900_sixx.source.toml`;
- `config/robots/tcp_tool0.json`;
- `pixi.lock` ve locked çalışma komutları.

F0-01 exact release snapshot'ını hash doğrulamasıyla edinmeli, Xacro'yu kilitli araçla çözmeli, `assets/robots/robot_a/manifest.json` üretmeli ve tam dosya/URDF/TCP/robot-spec hash zincirini T-F01 ile sınamalıdır. Uyuşmazlıkta geometri uydurulmaz; görev `FAIL`/`ENGELLİ` kalır veya robot değişikliği yeni ADR ile alınır.

## Koşu sonrası depo durumu

Bu bölüm koşu sırasındaki yukarıdaki gerçek zaman, başlangıç HEAD'i ve kirli çalışma ağacı kaydını değiştirmez; kapanıştan sonra doğrulanan Git durumunu ayrı tutar.

- F0-00 koşusunun başlangıç HEAD'i: `9b51aefcc6c1e87a7be36c8c0b055c91144ee86d`.
- F0-00 kapanış ve push commit'i: `ae9054c514c81b5ce237f89c86a9319b202c8745`.
- Remote durumu: kapanış commit'i `origin/main` üzerine pushlandı.
