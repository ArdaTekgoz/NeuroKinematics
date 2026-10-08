# C1-05 gerçek komutlar ve onay sonrası sıra

8 Ekim 2026 · r1. Depo kökü PowerShell. `commands/*/command.json` gerçek argv,
cwd, UTC, exit ve thread env; aynı dizindeki stdout/stderr ham baytları içerir.
Kayıt aracı: `.venv/c103/Scripts/python.exe scripts/c105_command.py --name <yeni-ad> -- <argv>`.
Aynı ad tekrar kullanılamaz. Her bağımlı komut bir önceki exit0 sonrası koşar.

## Gerçekleştirilen doğrulamalar

```powershell
git fetch origin main
git rev-parse HEAD main origin/main
.venv/c103/Scripts/python.exe scripts/check_c104_stage1.py --check
pixi run --locked .venv/c103/Scripts/python.exe scripts/audit_c105_inputs.py
pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_04 tests/c1_02/test_tc07_acceptance.py tests/c1_03 --mutation-output=experiments/C1-05/regression/mutations --junitxml=experiments/C1-05/regression/junit.xml
pixi run --locked .venv/c103/Scripts/python.exe -m neurokinematics.core.torch_validation --smoke --output experiments/C1-05/regression/fk-smoke
pixi run --locked .venv/c103/Scripts/python.exe -m neurokinematics.core.torch_validation --output experiments/C1-05/regression/fk-full --smoke-witness experiments/C1-05/regression/fk-smoke
```

`input-access` ilk denemesi exit1: audit kodu tek-pose `predict` API'sine batch
verdi. `input-access-v2` aynı özgün batch1024 inference protokolüyle düzeltildi,
exit0; eski dosyalar/sonuçlar/eşikler değişmedi. Başarısız komut logu korunur.
Audit scripti çıktı varsa overwrite etmez; tekrar kanıt üretmek için yeni kayıt
sürümü gerekir. Salt okunur tekrar için aşağıdaki `--check` kullanılır.

## Dondurma ve salt okunur tekrar

```powershell
.venv/c103/Scripts/python.exe scripts/check_c105_stage1.py --self-test
.venv/c103/Scripts/python.exe scripts/check_c105_stage1.py --freeze
.venv/c103/Scripts/python.exe scripts/check_c105_stage1.py --check
```

`--freeze` yalnız bu Aşama1 sonunda bir kez çalıştırılır. Sonraki denetimlerde
`--check`; drift varsa manifest yeniden yazılarak kabul edilmez. `SHA256SUMS`
dondurma öncesi tamamlanmış loglar dahil paket snapshot'ıdır. Kendisini veya
dondurma/sonraki audit/Git komutlarının gelecekte oluşan loglarını hashlemez.
Bu sonraki komutlar append-only saklanır, logger SHA'ları `--check` tarafından
ayrıca denetlenir. Git commit kimliği ayrıca bütün tracked paketi bağlar.

## Onay sonrası uygulanacak komut planı — NOT_RUN

1. Açık onayı Stage1 commit+SHA256SUMS kimliğine bağla; `--check` yeniden çalıştır.
2. Aynı kilitli overlay'de `pixi run --locked .venv/c103/Scripts/python.exe -m pip check`
   ve mevcut C1-03 artifact audit; eski C1-03 smoke/full+regresyonu yeni evidence
   köklerinde çalıştır. Girdi erişimini yeniden doğrula.
3. ADR-013 eğitim FK uzantısı, bağımsız oracle, kayıp ve negatif kontrolleri
   uygula. Planlanan yeni `scripts/run_c105.py` CLI henüz mevcut değildir.
   Geliştirme sonrası gerçek `--help`, altkomut/argümanlar ve kod SHA'ları ayrı
   Stage2 COMMANDS kaydında dondurulur; burada çalıştırılmış komut gibi verilmez.
4. Önce80 domain örneği/analitik/FD/gradcheck/mutant ve pilot kapıları; sonra
   E-C03 Q/FK üç seed, ardından E-C04 ve koşullu E-C05 kendi çiftleriyle.
5. Her run için tam per-row validation, gradyan/epoch, best/last metadata ve
   bağımsız Pinocchio ölçümünü hashle. Bütçe/seed/order/etiket/invalid payda audit.
6. Temiz checkout'ta checkpoint load/inference ve kanıt audit; taze ortamda
   eğitim tekrarı yoksa açıkça NOT_RUN. Model kararı, T-C04 ve C1-06 devir kaydı.

Onay öncesi yeni FK uygulaması/pilot/eğitim, T-C04 PASS veya C1-06 test açılışı yok.
