# Deney veya uygulama kaydı

Kimlik: RUN-20261010-C106R-D7
Durum: COMPLETE_DIAGNOSIS / VALIDATION_TARGET_NOT_MET
Görev ve gereksinim: C1-06R,REQ-C02–05; yerel ölçek kontrolü ve kıstas denetimi
Tarih ve sorumlu: 10 Ekim 2026; kullanıcının açık isteğiyle Codex

## Soru ve değişiklik

Tanı6'nın aynı yön verisi üzerinde sadece göreli pose ilk7 bileşenini
train-only standardize etmek local genellemeyi iyileştirir mi? Ardından
başarısızlığın ölçüt hesabı, hedef tanımı veya öğrenme yönüyle ilişkisi
incelendi. ADR-022/config, yeni model ön işleme/trainer,7 yeni test ve
bağımsız audit. Önceki kaynak/normalizasyon/veri/çıktılar değişmedi.

## Tekrar üretim

Başlangıç commit231613f, branch codex/c1-06r. Kullanıcının önceden var olan
STATUS/TRACEABILITY değişiklikleri ve untracked rapor/PDF/AUDIT korunur.
Pixi --locked + .venv/c106r, Python3.12.14, Torch2.10.0+cu128/CUDA12.8,
Windows/RTX5060 Laptop. Önceki cihaz kaydı Ryzen7 250/RAM24GB;
runtime audit.json'da. CPU kitaplık thread1,CUBLAS_WORKSPACE_CONFIG=:4096:8.
122 frozen kaynak/robot/TCP/veri/kilit hash kontrolü; yeni kaynak ve
directions/probe/provenance SHA'ları registration.json. Scaler yalnız o
ölçekteki train satırlarından; count,mean,std,pair_ids n64/n512-scaler.json.
Her checkpoint scaler SHA'sını da taşır. Seed2026100901,5000 adım,
fullbatch512/4096; model/optimizer/decode sözleşmesi config/ADR'de.

Gerçek komutlar depo kökünden:

```powershell
python scripts/c106r_command.py 053-local-feature-scale-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/c1_06r -q --junitxml=experiments/C1-06R/diagnostic7/tests.xml
python scripts/c106r_command.py 054-local-feature-scale-pair -- pixi run --locked .venv/c106r/Scripts/python.exe -m neurokinematics.neural.c106r_feature_scale
python scripts/c106r_command.py 055-feature-scale-validation-audit -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_c106r_feature_scale.py
```

UTC05313:57:08–13:57:17;05413:57:24–13:58:55;05513:59:23–13:59:35;
Türkiye+3. Eğitim komutu90,81s,20000 update; toplam46.080.000 maruziyet.
Hücre wall_s17,63/17,28/24,18/24,04. Komut/log SHA ve exit0 kaydı
experiments/C1-06R/commands/053–055 altında.

## Test ve ham kanıt

053:64PASS,0fail/skip; mevcut57+7 yeni.054:4koşu COMPLETE; RAW iki
kontrol önceki tensor/metrik birebir,4checkpoint bütün dört kümede replay.
055:33984 satır bağımsız FK/atan2 A/B eşliği,122 frozen ve scaler replay
PASS. Validation tanık çözümleri A/B3600/3600;1800 root,351 eksik wide
teacher dahil. Normalizasyon ile label/probe/validation fitting sızıntısı yok.
Oracle asla train girdisi değildir; öğrenilmiş model başarısı diye sayılmaz.

Ham checkpoint/tahmin yolları data/generated/C1-06R/diagnostic7/n64/n512-RAW/
LOCAL_Z altındadır; kesin yollar ve SHA'lar dört hücre JSON'unda.
validation-root-witness.json aynı raw kökünde; SHA audit.json'da.
Deney sonrası açıklayıcı component-range kapsamı ve q_current sabit referansı
audit içinde posthoc adıyla ayrıldı; yeni model seçimi/optimizasyon yapılmadı.
Eşik duyarlılığı0.5/1/2/5/10 çarpanları ön kayıtlı, kabul eşiği değişmedi.
Tüm repo regresyonu/Foundations üretimi/Docker/gerçek robot NOT_RUN;
tanı6 kinematik kanıtı exact-byte yeniden kullanıldı. Eski final NOT_READ,
yeni final NOT_CREATED.

## Sonuç ve yorum

Train A105→259/512 ve3→60/4096. Aynı-kök probe ve validation A/B0;
512-kök probe medyan13,79mm/4,34°→20,49mm/4,89°, local validation
16,04mm/4,84°→23,13mm/5,59°. Ölçekleme train'i iyileştirip genellemeyi
kötüleştirdi. Ölçüt hesabında kusur bulunmadı; ürün hedefi/araştırma kapanışı
ve local tanı/50-50 ürün kapsamı ayrımı kaynak raporlardan teyit edildi.
İki birincil yayın yeniden incelendi; eşdeğer olmayan literatür yüzdeleri
ürün hedefimizin kolaylığı için kanıt kabul edilmedi. Ayrıntı:
[RESULTS](RESULTS.md),[kıstas/gidişat](CRITERION_AND_DIRECTION_REVIEW.md).

## Sonraki adım

LOCAL_Z ile uzun eğitim yok; öğrenilmiş sıfır-düzeltme/yerel türev davranışı
ön kayıtlı tanı önerisi NOT_RUN. Model/geometri değişikliği henüz yapılmadı.
STATUS/TRACEABILITY güncellendi; kaynak Core §6'ya göre negatif araştırma
kapanışı mümkün, fakat bu turn C1-07 veya G1'i kapatmaz. Ürün NOT_MET.
