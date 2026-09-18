# F0-02 gerçek komut kaydı

18 Eylül 2026. Raw stdout/stderr, UTC başlangıç/bitiş ve çıkış kodları:
[`commands.json`](commands.json). Geliştirme sırasında başarısız denemeler:
[`development-attempts.json`](development-attempts.json). Kullanıcının izlenmeyen
geçici Word dosyası hiçbir stage/commit kapsamına alınmadı.

## Ön koşul ve geliştirme

| Gerçek komut / işlem | Çıkış | Sonuç |
|---|---:|---|
| `git branch --show-current` | 0 | `main` |
| `git fetch origin` | 0 | Remote güncellendi |
| `git rev-list --left-right --count main...origin/main` | 0 | `0 0`; pull gerekmedi |
| `git merge-base --is-ancestor 4048c428afceaab4418d6107897dcd36c2d48f33 origin/main` | 0 | F0-01 uygulama mevcut |
| `git merge-base --is-ancestor 1e0131122047aa8992601c7bb6e0cc94525bd709 origin/main` | 0 | F0-01 kapanış mevcut |
| `git status --short --branch` | 0 | Başlangıçta yalnız izlenmeyen Word dosyası |
| `Get-FileHash` ile dört immutable girdi | 0 | Beklenen SHA-256 değerleriyle aynı |
| `pixi run --locked fk-inspect` | 0 | 8 path joint, 6 aktif; isimden joint/frame eşlemeleri |
| `pixi run --locked test-f02-unit` | 1, 1, 0 | Analitik base tersleme düzeltmesi; 97/1, 97/1, 98/0 |
| `pixi run --locked validate-fk-small` | 1, 0 | Ortam metadata düzeltmesi sonrası 1000-q smoke hatasız; nihai kapı için INCONCLUSIVE |
| `pixi run --locked python scripts/run_f02_acceptance.py` | 1 | İlk koşu cp1254 print hatasında kesildi; ilk üç alt komut exit 0 |
| `pixi run --locked python scripts/run_f02_acceptance.py` | 0 | İlk tam gate; 99/99 F0-02 |
| `pixi run --locked python scripts/run_f02_acceptance.py` | 0 | Son gate; aşağıdaki frozen sıra, 102/102 F0-02 |

## Son frozen kabul koşusu

21:43:27–21:43:43 Europe/Istanbul. Bu sıra runner tarafından gerçekten yürütüldü;
ilk başarısız alt komutta durur. Nihai 10000-q testinde skip/slow dışlaması yok.

| Komut | Çıkış | Sonuç |
|---|---:|---|
| `pixi lock --check` | 0 | Lock güncel |
| `pixi install --locked` | 0 | Frozen kurulum |
| `pixi run --locked test-f00 --junitxml=experiments/F0-02/f00-junit.xml` | 0 | 6 passed, 0.37 s |
| `pixi run --locked verify-robot-a` | 0 | Immutable F0-01 kontrolü PASS |
| `pixi run --locked test-f01 --junitxml=experiments/F0-02/f01-junit.xml` | 0 | 16 passed, 0.68 s |
| `pixi run --locked test-f02-unit --junitxml=experiments/F0-02/unit-junit.xml` | 0 | 101 passed, 1.49 s |
| `pixi run --locked validate-fk` | 0 | 10000-q PASS, sıfır aşım |
| `pixi run --locked test-f02 --junitxml=experiments/F0-02/pytest-junit.xml -o junit_family=legacy` | 0 | 102 passed, 5.14 s; 10000-q yeniden hesaplandı |

Pixi'nin iç içe aktivasyonda ürettiği boş `SSL_CERT_DIR` uyarısı ham loglarda
korundu. Lock/install ve testler exit 0; sertifika ayarı veya güvenlik politikası
değiştirilmedi. `junit_family=legacy`, kabul testindeki sample hash ve maksimum
hataların JUnit properties olarak uyarısız kaydını sağlar.

## Kanıt bütünlüğü ve Git

- `pixi run --locked python -c "from scripts.run_f02_acceptance import hashes; hashes()"`: exit 0; geliştirme denemeleri kaydı dahil SHA256SUMS yenilendi.
- `pixi run --locked python -` ile 37 SHA256SUMS satırının raw bayt doğrulaması ve `load_robot()` kontrolü: exit 0, bütün hashler aynı.
- `git diff --check` ve `git diff --cached --check`: exit 0.
- `git diff --cached --stat`, `git diff --cached --name-only`, `git status --short`, `git log --oneline -n 5`: kapsam gözden geçirildi.
- `git commit -m "feat(foundations): implement F0-02 independent forward kinematics"`: exit 0; `d92dd213bb96f8932bd0019541dd13dd7365afaf`.

Kapanış ve push komutlarının sonucu, bu dosyayı içeren ikinci commit oluşturulup
normal `git push origin main` çalıştırıldıktan sonra kullanıcı yanıtında bildirilir.
Henüz yürütülmemiş Git adımları bu tarihsel komut tablosunda başarılı gösterilmez.
