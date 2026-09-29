# C1-03 Aşama 2 gerçek komutlar ve tekrar üretim

29 Eylül 2026 · matematik r1 / runtime ve harness r2. Tüm shell komutları repo
kökünde çalışır. `commands/*/command.json` exact argv, cwd, UTC ve exit kodunu;
aynı dizindeki stdout.log/stderr.log gerçek subprocess baytlarını saklar.
Komut kaydedici dört BLAS/OpenMP thread değişkenini 1 yapar, validator Torch'u 1 yapar.

## Kanonik kurulum

Foundations `pixi.lock` değişmez. Yeni checkout içinde:

```powershell
pixi install --locked
pixi run --locked python -m venv --system-site-packages .venv/c103
pixi run --locked .venv/c103/Scripts/python.exe -m pip install --ignore-installed --require-hashes --no-deps -r experiments/C1-03/requirements-win-cpu.lock
pixi run --locked .venv/c103/Scripts/python.exe -m pip install --ignore-installed --require-hashes --no-deps -r experiments/C1-03/stage2/runtime-supplement.lock
pixi run --locked .venv/c103/Scripts/python.exe -m pip check
pixi run --locked .venv/c103/Scripts/python.exe scripts/audit_c103_artifacts.py
```

`--ignore-installed` yalnız ayrı venv içine exact wheel'leri yazar; aynı version
string'ine sahip conda artifact'ini eş saymaz. Üçüncü taraf paketlerden 11/11
version/URL/SHA ve overlay yolu doğrulandı. İlk kurulumun eksik setuptools ve
OpenMP çakışması ADR-012'de; duplicate-runtime bypass kullanılmadı.

## Gerçek yerel sıra

| Komut kimliği | İş / sonuç |
|---|---|
| 001 | Stage1 checker --check, exit0 |
| 002–004 | venv0, Torch subprocess install0 (logger yazdırması ayrıca hata), pip-check1 |
| 005–007 | erken analitik abort3; izole import probe OMP#15; runtime supplemental resolution0 |
| 008–011 | supplemental0; local wheel override0; NumPy/Torch/Pinocchio matmul/import0; pip-check0 |
| 012–013 | analytic18PASS/1FAIL (tarihsel eager import); harness düzeltmesi sonrası19PASS |
| 014–016 | ilk r2 smoke/full0 ve100 unit/mutation PASS |
| 017–022 | observed dtype/input evidence audit sonrası smoke/full0,110 unit PASS, F0-01/02/03 16/102/159 PASS |
| 023–024 | typing_extensions exact wheel'in venv'e kurulması;11/11 artifact audit PASS |
| 025–027 | final exact overlay smoke/full0,110 unit/mutation PASS |
| 028 | yeni checkout/ortamda tam clean driver0,15 alt komutun tamamı0 |
| 029–031 | exact overlay F0-01/02/03 yeniden16/102/159 PASS |
| 032–033 | iki koşu/JUnit/source/artifact/input/raw kapanış audit0 |

Kanonik birincil matematik kanıtı `full-exact-a/`, ikincil `clean-b/full/`.
Önceki geçerli koşular ve başarısız denemeler ayrı dizinlerde korunur; kanonik
sonuçlara karıştırılmaz. Source mutant diff ve killing-test kayıtları
`mutations-exact-a/` ve `clean-b/mutations/`; ortak pytest exit/stdout/stderr
`027-exact-unit` ve `clean-b/commands/011-unit-mutation` içindedir.

## Sayısal kabul ve salt okunur audit

```powershell
pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_03/test_analytic.py
pixi run --locked .venv/c103/Scripts/python.exe -m neurokinematics.core.torch_validation --smoke --output temp/c103-next-smoke
pixi run --locked .venv/c103/Scripts/python.exe -m neurokinematics.core.torch_validation --output temp/c103-next-full --smoke-witness temp/c103-next-smoke
pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_03 --mutation-output=temp/c103-next-mutations --junitxml=temp/c103-next-unit.xml
pixi run --locked .venv/c103/Scripts/python.exe -m neurokinematics.core.torch_validation --verify --output experiments/C1-03/stage2/full-exact-a
```

Yeni output dizinleri mevcut olmamalı. F0-01/02/03 regresyonları ayrı pytest
çağrılarıyla, JUnit yeni output'a yazılarak çalıştırılır. C1-02 arayüz21 test
`tests/c1_03/test_pair_interface.py` içinde; büyük shard veya teacher etiketi gerekmez.
Tüm fonksiyonlar gerçek C1-02 üretim/validation kodunu çağırır.

## Gerçek temiz tekrar

Uygulama commit'i `7d9e282`; exact overlay audit düzeltmesi
`4022e2359306a780422c94f25252bd2eaa90ed8f`. İkinci checkout bu commit'te detached,
başlangıç tracked/untracked temiz, `.pixi` ve `.venv` yoktu (`clean-start.json`).

```powershell
# Yeni, temiz checkout'un kökünde; --output checkout DIŞINDA ve yeni olmalı.
python scripts/reproduce_c103.py --output C:/path/to/new-c103-evidence
```

Gerçek checkout: `C:/Users/Arda TEKGÖZ/.codex/worktrees/c103-clean-reproduction/NeuroKinematics-main`.
Ortama dosya kopyalanmadı; paket cache'i kullanılabilir. Gerçek çıktı bu reponun
`experiments/C1-03/stage2/clean-b/` dizinidir. Driver tam15 komut, kurulum,
artifact kontrolü, analitik, smoke, full, mutation/unit, üç regresyon ve evidence
audit'i fail-fast çalıştırdı. Başlangıç/bitiriş UTC `028-clean-driver` kaydında.

## Satır sonu ve teslim

Stage1 kanıtları değişmedi; eski checker Stage1 döneminin tam ağaç kontrolüdür.
Stage2 `load_contract` Stage1 SHA manifestini sabit SHA ile doğrular; değişen
durum belgelerinin **onaylı tarihsel Git blobunu**, immutable matematik girdilerin
**güncel içeriğini** kontrol eder. Yeni Torch modülünü eski glob envanterine sokmaz.

Stage2 JSON/JSONL/MD/script metinlerinin Git/kanonik kimliği UTF-8/LF;
ilk Windows çalışma ağacında bazı metinler CRLF olabilir. Gerçek log/JUnit baytları
`.gitattributes -text` ile korunur. `evidence-manifest.json` her dosyanın
raw çalışma ağacı SHA'sını ve canonical-LF SHA'sını ayırır. `SHA256SUMS` canonical
LF SHA-256 içerir; Git object SHA-1'i değildir. Kendi SHA dosyası dışarıda,
manifest SHA listesine dahildir. `full*/hashes.json` ve JSONL hashleri gerçek
raw matematik kanıt baytlarını denetler. Ham sonuçlar Git'te, LOCAL_ONLY değildir.

Teslim manifesti `python scripts/manifest_c103.py --write` ile üretilir.
`python scripts/manifest_c103.py` kanonik hashleri, tam dosya envanterini ve
log/JUnit raw hashlerini doğrular. `--verify-working-bytes` eklenirse tüm
dosyalarda özgün Windows snapshot baytları da zorunludur. Satır sonu dönüşümü
yalnız CRLF → LF'dir; başka whitespace değişikliği uygulanmaz. Aşama 1 ve
Aşama 2 kanıtları, yeni kaynak/test/script'ler ve değişen izlenebilirlik belgeleri
manifesttedir. Manifest kendisini ve SHA256SUMS'u içermez; SHA listesi
manifesti içerir ve yalnız kendisini dışlar.

Son kayıtların commit'i sonrası `git push origin main`; force yok. `git rev-parse
HEAD` ve `git ls-remote origin refs/heads/main` eşliği son kullanıcı yanıtında
bildirilir. Rapor kendi commit SHA'sını içine koyarak hash döngüsü oluşturmaz.
