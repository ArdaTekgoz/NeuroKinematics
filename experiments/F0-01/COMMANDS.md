# F0-01 gerçek komut kaydı

Tarih: 18 Eylül 2026. Aşağıdaki çıkış kodları bu çalışma sırasında gözlenen gerçek sonuçlardır. Kullanıcıya ait `~$uroKinematics_Model_Kullanim_Plani_r1.docx` hiçbir komuta dahil edilmedi.

## Başlangıç ve kaynak edinme

| Komut / işlem | Çıkış | Sonuç |
|---|---:|---|
| `git branch --show-current` | 0 | `main` |
| `git status --short --branch` | 0 | `main...origin/main`; yalnız kullanıcıya ait izlenmeyen Word lock dosyası |
| `git fetch origin` | 0 | Remote güncellendi |
| `git rev-list --left-right --count main...origin/main` | 0 | `0 0`; ayrışma/gerilik yok |
| `git merge-base --is-ancestor ae9054c514c81b5ce237f89c86a9319b202c8745 origin/main` | 0 | F0-00 kapanış commit'i remote üzerinde |
| `git clone --depth 1 --branch 2.0.2 https://github.com/kroshu/kuka_robot_descriptions.git <temp>/upstream` | NOT_CONFIRMED | Araç 30 saniyelik çıktı penceresini aştı; takip kontrolünde HEAD doğruydu fakat checkout yarım kalmıştı |
| `git -C <temp>/upstream rev-parse HEAD` | 0 | `fbda927964caa1eb4e408fb0c25fe46b5a0bde3c` |
| `git -C <temp>/upstream tag --points-at HEAD` | 0 | `2.0.2` |

İlk normal Windows checkout hash kontrolü yedi kritik dosyada CRLF dönüşümü nedeniyle sözleşmeyle uyuşmadı. Bu kabul sonucu sayılmadı. Snapshot, `git -c core.autocrlf=false checkout-index` ile raw Git blob baytlarından yeniden dışa aktarıldı; yedi kritik SHA-256 değeri sözleşmeyle eşleşti.

## Bağımlılık, üretim ve ara hatalar

| Komut / işlem | Çıkış | Sonuç |
|---|---:|---|
| `pixi lock` | 0 | `xacro 2.1.1` ve transitif `PyYAML 6.0.3` lock'a eklendi |
| `pixi install --locked` | 0 | Locked Windows ortamı kuruldu |
| `pixi run --locked build-robot-a` | 0 | İlk URDF/spec/manifest üretildi |
| `pixi run --locked verify-robot-a` | 1 | Xacro banner'ındaki rastgele temp yolundan ötürü deterministik yeniden üretim reddedildi |
| art arda iki `pixi run --locked build-robot-a` | 0, 0 | Sabit XML serileştirmesi sonrası URDF hashleri aynı |
| `pixi run --locked verify-robot-a` | 1 | Pinocchio dosya-yolu API'si checkout yolundaki `Ö` karakterini bozuk kodladı |
| `pixi run --locked verify-robot-a` | 0 | Aynı kalıcı UTF-8 URDF baytları Pinocchio XML API'siyle parse edildi; 29 dosya, 14 mesh, 6 joint doğrulandı |
| `pixi run --locked python -m neurokinematics.robot_asset verify-sources --output experiments/F0-01/source-verification.json` | 0 | Exact release/commit ve 7 kritik kaynak hash'i PASS |
| `pixi run --locked test-f01 --junitxml=experiments/F0-01/pytest-junit.xml` | 0 | `16 passed in 0.68s` |

## Kapanış kabul komutları

İlk toplu kapanış koşusu `20:07:55+03:00`'te başladı; `verify` fonksiyonuna kaynak raporu eklenirken oluşan yerleşim hatası nedeniyle `test-f01` 15 PASS / 1 FAIL verdi ve kabul edilmedi. Fonksiyon sınırı düzeltildikten sonra aşağıdaki frozen koşu yapıldı.

Kabul zamanı: `2026-09-18T20:08:35.4924021+03:00` – `2026-09-18T20:08:39.0257895+03:00`.

| Komut | Çıkış | Ölçülen sonuç |
|---|---:|---|
| `pixi lock --check` | 0 | Lock güncel |
| `pixi install --locked` | 0 | Frozen kurulum başarılı |
| `pixi run --locked test-f00` | 0 | 6/6 PASS, 0.27 s |
| `pixi run --locked build-robot-a` | 0 | URDF/spec/manifest üretildi |
| `pixi run --locked verify-robot-a` | 0 | 7 kritik kaynak, 29 dağıtılan dosya, 14 mesh, 6 joint ve Pinocchio nq/nv=6 PASS |
| `pixi run --locked test-f01 --junitxml=experiments/F0-01/pytest-junit.xml` | 0 | 16/16 PASS, 0.61 s |

Kabul koşusunda raporlanan hashler: URDF `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96`; robot-spec `4f97a2059d68a9b14fce50aed63628f3e664950033276b75c6a2cebd979ed95d`; manifest `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1`; TCP `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18`.
