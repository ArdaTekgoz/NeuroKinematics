# Deney veya uygulama kaydı

Kimlik: RUN-20261010-C106R-D8
Durum: COMPLETE_DIAGNOSIS; mevcut ürün sonucu NOT_MET
Görev ve gereksinim: C1-06R,REQ-C02–05; öğrenilmiş yerel tepki denetimi
Tarih ve sorumlu: 10 Ekim 2026; kullanıcının onayıyla Codex

## Soru ve değişiklik

Sabit model sıfır hedef farkında hedefi koruyor mu; küçük hedef değişimleri
doğru inverse türevi üretiyor mu? ADR-023/config, yeni tanı modülü,4 test ve
audit eklendi. Model/scaler/veri/eğitim veya eski kinematik kod değiştirilmedi.
K-I yalnız aynı dal beklentisi olarak, asıl task türevi prediction Jacobian
ile ölçüldü; alternatif geçerli IK dalı yanlış başarısız sayılmadı.

## Tekrar üretim

Başlangıç commit2cc29d7, branch codex/c1-06r. Kullanıcının STATUS/TRACEABILITY
ve untracked PDF/rapor/AUDIT değişiklikleri korunur. Windows,Python3.12.14,
Torch2.10.0+cu128/CUDA12.8,RTX5060 Laptop. Ryzen7 250/RAM24GB önceki cihaz
kaydı; runtime audit.json'da. Pixi --locked/.venv/c106r;CPU kitaplık thread1,
CUBLAS_WORKSPACE_CONFIG=:4096:8. Robot/TCP/veri/runtime kaynakları122 frozen
SHA ile, yeni kaynaklar ve dört checkpoint/scaler registration.json ile
doğrulandı. Modellerin eğitim seed'i2026100901; bu tur yeni eğitim seed'i veya
update yok. Sıfır grupları64/512 train+1800 validation kökü ×2 anchor;
türev seçimi ortak32 train+32 main validation ×2 anchor. h üç değerde sabit.

Gerçek komutlar depo kökünden:

```powershell
python scripts/c106r_command.py 056-local-response-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/c1_06r -q --junitxml=experiments/C1-06R/diagnostic8/tests.xml
python scripts/c106r_command.py 057-fixed-local-response -- pixi run --locked .venv/c106r/Scripts/python.exe -m neurokinematics.neural.c106r_local_response
python scripts/c106r_command.py 058-local-response-audit -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_c106r_local_response.py
```

UTC05614:24:16–14:24:26;05714:24:48–14:25:12;05814:26:25–14:26:47;
Türkiye+3.057 komutu24,48s, modül ölçümü19,73s;05822,09s.
Komut/log/exit0 SHA kayıtları experiments/C1-06R/commands/056–058 altında.

## Test ve ham kanıt

056:68PASS,0fail/skip;4 yeni test ideal/hareketsiz/ters-işaret kontrolleri,
feature tangent FD/frame mutantı, zero/tiny oracle ve FD axis sırası.
057:128 anchor preflight (256 current/teacher FK-Jac eşlik satırı,128 FD)
PASS.512 model-anchor shadow64 FD/autograd C1-03 atol1e-5/rtol1e-3 PASS.
16704 sıfır ve18432 küçük-hedef sorgusu;058 bütün prediction/metric ve
matrisleri yeniden üretti.122 frozen/hash ve ağırlık state hash değişmez.

Ham veri: data/generated/C1-06R/diagnostic8/{model}/zero.json ve
train-root/train-current/validation-root/validation-current.npz. Tahmin,
hedef,anchor,K,J,teğet veFD matrisleri saklandı; tam yollar/SHA'lar hücre
JSON'larında. Yön kosinüsü/kolon normları audit'te posthoc olarak ayrıdır.
Sıfır ve küçük hedefler bağımsız FK'den üretilmiş ulaşılabilir tanık taşır.

Tüm repo/Foundations üretim regresyonu, Docker, fiziksel robot NOT_RUN.
Yeni optimizer/eğitim NONE. Eski final NOT_READ; yeni final NOT_CREATED.
Orijinal3600 validation yeniden kampanya olarak koşulmadı; ağırlıklar
değişmediğinden önceki NOT_MET kararını bu sentetik oranlarla değiştirmiyoruz.

## Sonuç ve yorum

512 RAW yeni-kök current zero sorguları A96/B7/1800; medyan5,61mm/1,25°.
512 LOCAL_Z A35/B4/1800;7,24mm/1,50°. Aynı current'ı koruyan referans
bütün zero sorgularda A/B sağlar. Yerel task türevi medyan sapması yeni-kök
current grubunda RAW5120,604,LOCAL_Z5120,732; kapsam31/32. Düşük-condition
yarısı sırasıyla0,621/0,764. Küçük h fp32 farkı ayrı ölçüldü; shadow64
FD farkı en çok yaklaşık2,46e-7. Büyük davranış sapması yalnız yuvarlama
değil. Bu mekanizmalar tek nedensel eğitim kusuruna indirgenmedi.
[Sonuç](RESULTS.md),[audit](audit.json).

## Sonraki adım

[Centered residual başlık önerisi](NEXT_EXPERIMENT.md) PROPOSED/NOT_RUN.
Sıfır ofseti yapısal olarak korunabilir; aynı ağırlıklarda merkezleme türevi
değiştirmez. Yeni eğitim/maliyet kontrolü ve ayrı ADR gerekir. Bu tur ürün
eşiği,model,eski kanıt değişmedi; C1-07/G1 başlatılmadı. STATUS/TRACEABILITY
ve teslim manifesti güncellendi; kullanıcı değişiklikleri commit dışında.
