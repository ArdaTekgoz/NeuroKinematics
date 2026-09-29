# C1-03 Aşama 1 komutları

Çalışma kökü repo kökü; Windows PowerShell. Aşağıdakiler bu oturumda gerçekten
çalıştırıldı. Başlangıç okumaları araç oturum kaydında; tekrar edilen son statik
komutlar exit/stdout/stderr/UTC ile `commands.json` ve `logs/` altında tutulur.

| Komut / eylem | Sonuç |
|---|---|
| `git status --short`, `git branch --show-current`, `git rev-parse HEAD`, `git remote -v`, `git log -5 --oneline` | main, b82faf95…, origin GitHub; yalnız iki ilgisiz untracked Word |
| `git ls-remote origin refs/heads/main` | b82faf95…; başlangıç local/remote eş |
| `Get-Content` / `rg` görev, rapor, kaynak, test, lock okumaları | Okundu; yanlış `configs`, `robots`, `docs/adr/ADR_TEMPLATE.md` yolları bulunamadı; gerçek `config`, `assets/robots`, `docs/templates/ADR_TEMPLATE.md` ile düzeltildi |
| `pixi --version` | 0.81.0 |
| `pixi run --locked python -c ...` (runtime metadata) | Python 3.12.14, NumPy 2.5.3, Pinocchio 4.1.0, pytest 8.4.2, Torch yok |
| Aşağıdaki pip dry-run | exit 0; dokuz paket; `dependency-resolution.json` + `.log`; kurulum yok |
| PyPI 0.10.0 metadata/wheel okuma | Hash ve MIT/API doğrulandı; `candidate-metadata.json`; runtime NOT_RUN |
| `python temp/c103/metadata.py` ilk deneme | exit 1; Jinja2/MarkupSafe index girdisinde hash yoktu; küçük wheel baytları indirilip SHA hesaplandı, eşik değişimi yok |
| Aynı metadata helper düzeltilmiş tekrar | exit 0; `dependency-pins.json`, `requirements-win-cpu.lock`; helper geçici, kaynak URL ve pinler kalıcı |
| Default Windows encoding ile JSON okuma denemesi | UnicodeDecodeError; açık UTF-8 ile düzeltildi; kaynak veri değişmedi |
| `pixi run --locked python scripts/check_c103_stage1.py --freeze` | exit 0; 15 G0 / 29 robot / 85 girdinin statik kontrolü, 1086 örnek; FK/gradient çağrısı yok |
| Sonlandırmada ilk `git diff --check` | exit 1; değiştirilmiş üç başlık satırındaki Markdown çift boşluk uyarısı düzeltildi; `logs/diff-check-first.*.log` korunur |

Gerçek dry-run komutu (host Python 3.11 pip 26.1.2, hedef CPython 3.12):

```powershell
python -m pip install --dry-run --ignore-installed --only-binary=:all: --platform win_amd64 --python-version 3.12 --index-url https://download.pytorch.org/whl/cpu --report experiments/C1-03/dependency-resolution.json 'torch==2.10.0+cpu'
```

Bu resolver komutu gelecekte yeniden seçim yapabilir; kuruluma esas olan exact
URL+SHA içeren mevcut `requirements-win-cpu.lock` dosyasıdır.
Metadata helper paket import etmedi; zip kaynaklarını okudu. Yeniden inceleme:
`candidate-metadata.json` içindeki wheel URL'sini indir, SHA-256'yı karşılaştır,
zip içindeki `pytorch_kinematics/chain.py` ve `urdf.py` hashlerini doğrula.

## Salt okunur tekrar

```powershell
pixi lock --check
pixi run --locked python scripts/check_c103_stage1.py --check
git diff --check
```

`--check` frozen örnekleri yeniden üretir ve dosya/robot/bağımlılık/config/SHA
bağlarını kontrol eder; FK/gradient/test suite başlatmaz ve kanıt yazmaz.
`--freeze` mevcut sample/manifest üzerine yazmayı reddeder; tekrar komutu değildir.
Ortam bilgisi orijinal Stage1 gözlemidir; Stage2 ayrı environment kaydı yazmalıdır.

## Aşama 2 — PLAN / NOT_RUN

Yalnız açık kullanıcı onayı ve Stage1 SHA kontrolünden sonra:

```powershell
pixi install --locked
pixi run --locked python -m venv --system-site-packages .venv/c103
pixi run --locked .venv/c103/Scripts/python.exe -m pip install --require-hashes --no-deps -r experiments/C1-03/requirements-win-cpu.lock
pixi run --locked .venv/c103/Scripts/python.exe -m pip check
```

Explicit dokuz paket `--no-deps` ile kurulacak, eksik/uyumsuz bağımlılık `pip check`
ve gerçek runtime sürüm/import kapısında durduracak. PATH/DLL ortamı Pixi'den
gelir. Bu komutlar henüz **NOT_RUN**, çalışır kurulum iddiası değildir.
Torch FK/test runner dosyaları henüz yok; uydurma kabul CLI komutu verilmez.
Onaydan sonra gerçek yollar COMMANDS'a eklenir ve matris sırası korunur.

Planlanan mevcut regresyon girişleri `tests/f0_01`, `tests/f0_02`, `tests/f0_03`;
C1-02 için TEST_MATRIX R02'deki yeni shard bağımsız `tests/c1_03` arayüz fixture'ı.
Mevcut `test_tc07_mutations.py` roots() ile F0-04 büyük shardlarını açtığı için
doğrudan seçilmedi. JUnit yeni C1-03 run dizinine yazılır.
F0/C1-02 kanıtları üzerine çıktı yazılmaz. İkinci fresh checkout/env de aynı
kilit/pinlerle kurulur. `pip check` başarısızsa kabul testine geçilmez.

## Git teslimi

Yalnız .gitattributes, C1-03 script/kanıtları, ADR ve ilgili görev/durum/trace/roadmap
dosyaları stage edilir. `git diff --cached --check` ve dosya/size kapsamı incelenir.
Commit: `docs(core): freeze C1-03 differentiable FK contract`.
Push: `git push origin main`; force yok. Sonra `git rev-parse HEAD` ve
`git ls-remote origin refs/heads/main` eşitliği kontrol edilir. Gerçek teslim
SHA ve push sonucu son kullanıcı yanıtında; belge kendi commit SHA'sını içermez.
