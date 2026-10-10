# Deney veya uygulama kaydı

Kimlik: RUN-20261010-C106R-D9
Durum: COMPLETE_DIAGNOSIS / CONTINUATION_GATE_FAIL / PRODUCT_TARGET_NOT_MET
Görev ve gereksinim: C1-06R,REQ-C02–05; hibrit önceliğe göre araştırma sınırı
Tarih ve sorumlu: 10 Ekim 2026; kullanıcının onayıyla Codex

## Soru ve değişiklik

Sıfır hedef farkında residual'ı merkezlemek aynı-kök ve yeni-kök genellemeyi
iyileştiriyor mu? ADR-024/config eğitimden önce kaydedildi; yeni modül,
4 test ve bağımsız audit eklendi. Mevcut ağırlıklar, veri, FK, eşikler ve
Foundations değiştirilmedi. Yeni kayıtlı CENTERED başlık aynı current için
g(x)-g(x_zero) kullanır. RAW karşılaştırması üç eşli seed ile yürütüldü.
Kullanıcının açık uzun vadeli tercihi güvenilir hibrit IK olarak kaydedildi.

## Tekrar üretim

Başlangıç commit f5f1036, branch codex/c1-06r. Kullanıcının STATUS/TRACEABILITY
ve untracked PDF/rapor/AUDIT değişiklikleri korunur ve commit dışında kalır.
Windows11 build26200;Python3.12.14;Torch2.10.0+cu128/CUDA12.8;RTX5060 Laptop.
Ryzen7 250/24GB RAM önceki cihaz kaydı. Pixi --locked ve .venv/c106r;
CPU kitaplık thread1,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONUTF8=1.
Robot/TCP/veri/runtime122 frozen SHA; yeni kaynaklar, normalization, önceki
RAW checkpoint ve directions/probe verisi registration.json'da SHA ile.
Raw ağırlık/tahmin/matris yolları ve hashleri hücre JSON'larında; Git dışı.

Seed2026100901/02/03;512 train kökü,4096 directions/4096 ayrı probe;
eski RAW normalizasyonu;width512,AdamWlr.001wd.01,cosineeta1e-6;
her hücre5000 full-batch update,terminal checkpoint. Her seed'in iki kolu
aynı başlangıç tensor hash'ine sahip; üç seed farklı. Bütçe ve karar kapısı
sonuç görülmeden donduruldu. İki-forward centered maliyeti ayrıca ölçüldü.

Gerçek komutlar depo kökünden:

```powershell
python scripts/c106r_command.py 059-centered-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/c1_06r -q --junitxml=experiments/C1-06R/diagnostic9/tests.xml
python scripts/c106r_command.py 060-centered-training -- pixi run --locked .venv/c106r/Scripts/python.exe -m neurokinematics.neural.c106r_centered
python scripts/c106r_command.py 061-centered-audit -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_c106r_centered.py
```

059 UTC17:25:25,616–17:25:37,758;060 UTC17:25:58,528–17:29:10,221;
Türkiye+3.061 tam başlangıç/bitiş ve her komutun argv/env/exit0/stdout/stderr
SHA kaydı commands/059–061 altındadır. Eğitim komutu191,69s;
modül187,13s;72 test9,93s pytest ölçümü.

## Test ve ham kanıt

059:72PASS,0fail/error/skip. Yeni4 test sıfır girdisi/current koruma ve
mutasyonsuzluk, aynı ağırlıkta hedef türevi eşliği, finite difference/gradient
ve gerçek FK sıfır-hareket roundtrip kontrolüdür.
060:64 seçilmiş anchor FK/Jac/FD preflight,384 model-anchor shadow64 FD;
ilk RAW exact önceki tensor ve dört metrik seti. Altı model/six-set
weights_only reload eşliği PASS. Toplam30000 update/122880000 maruziyet.
061:87696 prediction checkpoint replay + bağımsız FK/atan2 sınıflandırması;
384 derivative anchor,shadowFD yeniden hesaplama ve matris replay PASS.
122 frozen/hash değişmez. Sıfıra yakın açıda acos/atan2 sayısal farkı için
audit toleransı5e-6°; A/B fiziksel eşikleri değişmedi. Gerçek maksimum
FK/açı farkları audit.json'da. Aynı satırların farklı model/set ölçümleri
bağımsız hedef sayısı diye sunulmadı.

Ham kanıt data/generated/C1-06R/diagnostic9/s{seed}-{arm}/last.pt,
predictions.json,train-derivative.npz,validation-derivative.npz.
Config/registration/preflight/results/audit/test XML ve komut logları
experiments/C1-06R altında takip edilir. Sealed eski final NOT_READ;
yeni bağımsız final NOT_CREATED. Tüm repo/Foundations regresyonu, Docker,
temiz yeni ortam, fiziksel robot ve hibrit refinement NOT_RUN.

## Sonuç ve yorum

CENTERED zero2312/2312 A/B her seed'de PASS; train A568/657/667,
probe A60/70/82. Validation A1/0/0, B0/0/0; kontrol A0/0/0.
Local konum medyanı16,0–16,5→19,8–20,9mm, limit ihlalleri arttı;
yeni-kök türevi de kötüleşti. Ön kayıtlı devam kapısı FAIL, mekanik sıfır
koruma PASS. Tek validation başarısını genel çözüm olarak yorumlamıyoruz.
Eşli-rootbootstrap%95 local ortalama kazanç aralığı[0;0,05556]yp;
validation araştırma kümesi, üç seed'e koşullu belirsizlik.
[Sonuç](RESULTS.md),[audit](audit.json).

## Sonraki adım

Bu MLP başlık/ölçek deney ailesi durduruldu; yeni uzun eğitim hazır değil.
[Araştırma yönü](RESEARCH_DIRECTION_AND_STOP_RULES.md) hibrit ürün hedefini,
bilinmeyen neural başarı tavanını ve sınırlı araştırma bütçesini açıklar.
C1-06R kapanış/devir kararı, ardından C1-07/T-C06/G1 önerilir; bu tur
bu görevler veya H1 başlatılmadı. C1-06R ürün hedefi NOT_MET kalır.
STATUS/TRACEABILITY/task ve teslim manifesti güncellenir; kullanıcı
değişiklikleri commit kapsamına alınmaz. Push NOT_RUN.
