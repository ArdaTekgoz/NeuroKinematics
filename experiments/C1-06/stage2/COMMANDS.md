# C1-06 Aşama 2 çalıştırma ve tekrar üretim

8 Ekim 2026. Stage1 `075478545a30373db2a4ac64434f2db678df050e`;
açılış öncesi yürütücü `e97425d` (tam SHA Git geçmişinde). Kullanıcı onayı
approval.json; kaynak/test hashleri preflight.json. Stage1 belgeleri tarihsel
mühürlü kayıttır, sonradan güncel durumla değiştirilmez.

Her komut depo kökünde, tek CPU/BLAS thread politikasını uygulayan
scripts/c106_command.py kaydedicisiyle çalışır. Gerçek argv, exit, UTC
başlangıç/bitiş, stdout/stderr SHA ve içerikleri ../commands altında.

```powershell
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-input-check -- .venv/c103/Scripts/python.exe scripts/check_c106_stage1.py --check
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-c105-witness -- pixi run --locked .venv/c103/Scripts/python.exe scripts/c105_witness.py verify --witness experiments/C1-05/stage2/fixed-validation-witness.json --output experiments/C1-06/stage2/c105-witness.json
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-c104-witness -- pixi run --locked .venv/c103/Scripts/python.exe scripts/c104_witness.py --check
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-runtime-tests -- pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_06 --junitxml=experiments/C1-06/stage2/runtime-tests.xml
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-preflight -- pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py preflight
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-identity -- pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py identity --run final-001
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-evaluate -- pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py evaluate --run final-001
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-summarize -- pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py summarize --run final-001
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-audit -- pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c106.py audit --run final-001
```

Komut listesinin varlığı çalıştırılma kanıtı değildir; command.json ve ilgili
gate/result çıktısı otoritedir. C1-04 tarihsel wrapper exit1: eski global
.gitattributes snapshot'ı; doğrudan 21-checkpoint preflight bunun yerine asıl
metadata/weight/config/robot ve on tanığı doğrular. Eski hata logu korunur.

Yeniden üretimde önce LOCAL_ONLY raw/checkpoint/shardları manifestteki göreli
yollara ayrıca taşı. Mutlak checkpoint yolları yerine C1-06 config göreli
yollarını kullan. Stage1 checker orijinal dosyaları/byte hashlerini kontrol
eder. Mevcut preflight ve kampanya dosyaları `open('x')` ile korunur; otomatik
overwrite yoktur. Yeni ölçüm için yeni çalışma alanı ve açıkça yeni run adı
kullan; eski dosyaları silme veya kabul edilen raw üzerine yazma. Aynı final
set tekrarlandığında bunun yeni bağımsız test olduğunu iddia etme.

Temiz ortam kurulumu C1-03/05 kayıtlarındaki locked Pixi + hashli Torch overlay
tarifine dayanır. C1-06 kapsamında yeni temiz ortam kurulumu veya eğitim
tekrarı NOT_RUN. Kaynak wrapper/kernel hashleri kontrol edilmeden eski kabul
sonuçlarını yeni ortama taşıma. Linux/GPU, fiziksel robot ve collision NOT_RUN /
NOT_CHECKED. Uzak büyük dosya arşivi NOT_CONFIRMED.

## 9 Ekim teslim bütünlüğü

```powershell
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-bundle-freeze -- .venv/c103/Scripts/python.exe scripts/check_c106_bundle.py --freeze
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-bundle-freeze-002 -- .venv/c103/Scripts/python.exe scripts/check_c106_bundle.py --freeze
.venv/c103/Scripts/python.exe scripts/c106_command.py --name stage2-bundle-check -- .venv/c103/Scripts/python.exe scripts/check_c106_bundle.py --check
```

Bu kontroller yeni model çalıştırmaz: önceki 322 girdiyi, Stage1 mührünü,
preflight kaynaklarını, 27 raw dosyasının byte/SHA/satır sayısını, 59 test
kaydını, karar/devir bağlarını ve 36.000 C1-04/Q eşli final çıktısını denetler.
Freeze küçük Stage2 kanıtını evidence-manifest.json ve SHA256SUMS ile sabitler;
sonraki check salt okunurdur. Freeze/check komutlarının kendi sonradan oluşan
logları döngüsel hash oluşturmamak için manifest dışında, command.json içindeki
SHA değerleriyle kontrol edilir. Gerçek çıkış durumu ilgili komut kaydıdır.

İlk teslim freeze denemesi exit1: query-diagnostics.json bir JSON dizisi
olmasına rağmen checker bütün raw manifest girdilerinde JSONL rows alanı
bekledi. Yalnız teslim checker'ı düzeltildi: 1.860.000 ölçüm satırı ve 12.000
tanı girdisi ayrı sayılır. İlk hata logu korunur; o denemede manifest
yazılmadı. Ölçüm/istatistik yürütücüsü ve kabul edilmiş sonuçlar değişmedi.
