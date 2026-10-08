# C1-06 komut ve uygulama sırası

Depo kökünde PowerShell; mevcut kilitli Pixi + .venv/c103 Python overlay.
Her çalışmada benzersiz --name; recorder mevcut komut klasörünü ezmez.
Gerçek argv, UTC zamanları, exit ve stdout/stderr SHA: commands/*/command.json.

## Bu Aşama 1'de çalıştırılanlar

```powershell
.venv/c103/Scripts/python.exe scripts/c106_command.py --name input-audit -- .venv/c103/Scripts/python.exe scripts/audit_c106_inputs.py
.venv/c103/Scripts/python.exe scripts/c106_command.py --name input-audit-002 -- .venv/c103/Scripts/python.exe scripts/audit_c106_inputs.py
.venv/c103/Scripts/python.exe scripts/c106_command.py --name synthetic-tests -- pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_06 --junitxml=experiments/C1-06/synthetic-tests.xml
.venv/c103/Scripts/python.exe scripts/c106_command.py --name synthetic-smoke -- pixi run --locked .venv/c103/Scripts/python.exe scripts/smoke_c106.py
.venv/c103/Scripts/python.exe scripts/c106_command.py --name synthetic-tests-final -- pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_06 --junitxml=experiments/C1-06/synthetic-tests-final.xml
```

İlk audit exit1, yanlış witness yolu düzeltildikten sonra exit0. İlk audit JSON'u
input-audit-attempt-001.json adıyla korundu; yeni audit ayrı logla yazıldı.
İlk 26 ve son 31 sentetik test ayrı JUnit/log kayıtlarıdır; toplanıp 57 bağımsız
test diye sunulmaz. Smoke on sentetik girdide yedi başarı ve üç bilerek hatalı
girdi üretir; bu sayılar T-C05 sonucu değildir.

## Ön kayıt kapanış sırası

```powershell
.venv/c103/Scripts/python.exe scripts/c106_command.py --name freeze -- .venv/c103/Scripts/python.exe scripts/check_c106_stage1.py --freeze
.venv/c103/Scripts/python.exe scripts/c106_command.py --name verify-frozen -- .venv/c103/Scripts/python.exe scripts/check_c106_stage1.py --check
git diff --check
```

Ardından yalnız C1-06 dosyaları ve bu görevin STATUS/TRACEABILITY ekleri commit
edilir; başlangıçtaki kullanıcı audit değişiklikleri commit dışında korunur.
Normal push sonrası commit ve uzak main SHA eşliği kontrol edilir. SHA256SUMS
kendini veya yazılmakta olan freeze komut logunu içermez. Sonraki komut logları
checker tarafından ayrıca doğrulanır; snapshot sessiz yeniden oluşturulmaz.

## Onaydan sonraki tek yönlü Aşama 2

Aşağıdaki sıra PLANLANDI / NOT_RUN. Kullanıcı onayı Stage1 commit/SHA ile
approval.json'a kaydedilir. İlk komut mevcut ve çalışabilir:

```powershell
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-input-check -- .venv/c103/Scripts/python.exe scripts/check_c106_stage1.py --check
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-c105-witness -- pixi run --locked .venv/c103/Scripts/python.exe scripts/c105_witness.py verify --witness experiments/C1-05/stage2/fixed-validation-witness.json --output experiments/C1-06/stage2/c105-witness.json
```

Devamındaki CLI adları **ayrılmış uygulama planıdır, henüz mevcut yürütücü
değildir**; çalıştırılmış veya hazır kabul edilmez. Aşama 1 talebi doğrultusunda
istatistik/karar taslağı src/neurokinematics/neural/c106.py ve sentetik testleri
hazırdır. Aşama 2 başında bu sıranın yürütücüsü tamamlanıp hashlenir:

```powershell
# PLANLANAN CLI; scripts/run_c106.py Aşama 2'de yazılacak.
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py preflight --config experiments/C1-06/config.json --approval experiments/C1-06/stage2/approval.json
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py identity --config experiments/C1-06/config.json --run final-001
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py evaluate --config experiments/C1-06/config.json --run final-001
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py summarize --config experiments/C1-06/config.json --run final-001
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py audit --config experiments/C1-06/config.json --run final-001
```

Her planlı komut aynı c106_command recorder ile ayrı ad/argv kaydında çalışır.
Preflight: C1-04 üç + C1-05 18 checkpoint load/10 witness, güncel sentetik
negatif testler, tam envanter ve runtime. Identity: onaylı tek açılış, query
soy/kimlik/FK ve beş baseline raw join; sonuç görülerek tuning yapılmaz.
Evaluate: configteki 21 checkpoint, üç seed, 12.000 sorgu, beş geçiş, tek CPU
thread. Summarize: tam payda, primary/secondary paired CI, seed varyasyonu,
hata ve süre tabloları. Audit: bütün raw/log SHA/sayıları ve karar; yalnız
eksiksizse T-C05, ardından C1-07-handoff.json. Bu adımların hiçbiri Aşama 1'de
çalıştırılmaz. Kesinti/teknik kusurda eski attempt korunur ve ADR açılır.
