# Deney veya uygulama kaydı

Kimlik: RUN-20261010-C106R-D6
Durum: COMPLETE_DIAGNOSIS / VALIDATION_TARGET_NOT_MET
Görev ve gereksinim: C1-06R; REQ-C02–05; C1-02R/v1 veri kapsamı araştırması
Tarih ve sorumlu: 10 Ekim 2026; kullanıcı onayıyla Codex

## Soru ve değişiklik

Aynı kökte yön çeşitliliği, tekrar maruziyetine karşı yerel IK öğrenmesini
iyileştiriyor mu? ADR-021, config, yeni generator/trainer, dört mutant/provenance
testi ve bağımsız audit eklendi. Tarihsel üretici/model/ölçüm kodu değişmedi.
Kabul A2mm/1°,B1mm/0.5° ve üç seed main A≥%95 ürün kapısı korunur.

## Tekrar üretim

Başlangıç commit d6a5a4b, codex/c1-06r. Önceden var olan STATUS/TRACEABILITY
düzenlemeleri ve untracked PDF/rapor/AUDIT dosyaları kullanıcıya ait; korunur.
Python3.12.14, Torch2.10.0+cu128, CUDA12.8; RTX5060 Laptop, Windows;
Ryzen7 250/RAM24GB önceki cihaz kaydı. Güncel runtime audit.json içinde.
Pixi --locked + .venv/c106r overlay; robot/TCP/veri/kilit122 SHA training-freeze
ile, yeni kaynaklar registration.json ile kontrol edildi. CPU kütüphane
thread1, CUBLAS_WORKSPACE_CONFIG=:4096:8; komut ortamı kayıtlarda.
Data seed2026101001, model seed2026100901; diğer bütçeler config.json/ADR-021.
checkpoint ve ham shard SHA'ları preflight.json ve hücre JSON'larında.

Gerçek komutlar depo kökünden:

```powershell
python scripts/c106r_command.py 050-local-directions-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/c1_06r -q --junitxml=experiments/C1-06R/diagnostic6/tests.xml
python scripts/c106r_command.py 051-local-direction-coverage -- pixi run --locked .venv/c106r/Scripts/python.exe -m neurokinematics.neural.c106r_directions
python scripts/c106r_command.py 052-local-direction-audit -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_c106r_directions.py
```

UTC başlangıç/bitiş:05013:44:50–13:45:07;05113:45:12–13:46:54;
05213:47:20–13:47:32. Türkiye saati+3. Eğitim/üretim komutu102,10s;
20000 optimizer update,46.080.000 toplam satır maruziyeti. Dört hücrenin
kendi ölçülen süreleri17,91/17,25/24,58/23,93s; preflight ayrıca zaman alır.

## Test ve ham kanıt

050:57 PASS,0fail/skip. Yeni4 test: seed/replay/kök, sınırlarda rejection,
validation/overlap/etiket mutantları, bağımsız göreli dönüşüm/etiket bağımsızlığı.
051: PASS teknik yürütüm;8192 yeni sürümlü satır,16384 FK/Jacobian,
teacher A/B8192/8192, aynı initialization ve eşli maruziyet,4checkpoint bütün
değerlendirme kümeleri reload PASS.052: PASS;122 frozen,8192 provenance/allfield
replay,33984 prediction bağımsız FK,ek1024 FK/Jac ve32 farklı kök FD.

Ham yollar: data/generated/C1-06R/diagnostic6/{directions,probe}.npz,
provenance.json ve n64/n512-REPEAT/DIRECTIONS altındaki last.pt/predictions.json.
Kesin yollar/SHA'lar preflight.json ve dört hücre JSON'unda. stdout/stderr,
komut zamanları ve exit0 kanıtı experiments/C1-06R/commands/050–052 klasörlerinde.
Tüm Foundations üretim/benchmark, Docker yeniden kurulum, fiziksel robot,
eski final ve yeni final test NOT_RUN. Eski final ham dosyaları okunmadı.

## Sonuç ve yorum

[RESULTS](RESULTS.md) sayısal tabloyu içerir. Yön müdahalesi sürekli hataları
azaltır; aynı-kök yeni yönler ve validation her hücrede A/B0.64 tekrar64/64
orijinal başarısı yeni yönlere aktarılmaz. Yön train A105/512 ve3/4096:
mevcut öğrenme tasarımında yerel hassasiyet de çözülememiştir. Bu veri kapsamı
tanısı başarılı bir ürün değildir; tek seed ve eşit güncelleme sınırı korunur.

## Sonraki adım

[NEXT_DIAGNOSTIC](NEXT_DIAGNOSTIC.md): aynı veri üzerinde train-only yerel girdi
ölçeği kontrolü PROPOSED/NOT_RUN. Uzun kampanya hazır değil; C1-07/G1 açık.
STATUS/TRACEABILITY ve teslim manifesti yeni kanıtla güncellendi; tarihsel
REPLAN ve önceki raporlar değiştirilmedi. Kullanıcı değişiklikleri commit dışı.
