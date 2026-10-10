# Deney veya uygulama kaydı

Kimlik: RUN-20261010-C107-PREP
Durum: COMPLETE_PREPARATION; C1-06R CLOSED_WITH_UNMET_PRODUCT_TARGET; G1 OPEN
Görev ve gereksinim: C1-07/REQ-C06 hazırlığı; C1-06R kapsam durdurma/devir
Tarih ve sorumlu: 10 Ekim 2026; kullanıcının ikinci aşama komutuyla Codex

## Soru ve değişiklik

Doğrudan IK başarısızlığı sonrasında hangi kanıt/model hangi kısıtlarla
hibrit araştırmaya hazırlanabilir? ADR-025 ile C1-06R araştırma durdurması,
ana FK_TANH ve ikincil local-only RAW üçer seed seçimi çıkarımdan önce
sabitlendi. Yeni devir manifesti, model kartı taslağı, G1 hazırlık matrisi,
checkpoint/CPU validation denetleyicisi ve komut kaydedici eklendi.
Ürün eşiği, eski H2 reddi, robot ve ağırlıklar değiştirilmedi.

## Tekrar üretim

Başlangıç commit2ab984eb32808a5d86e9625905a3e14d390156c1;
branch codex/c1-06r, birinci aşamada origin'e push doğrulanmıştı.
STATUS/TRACEABILITY ve untracked PDF/AUDIT/rapor kullanıcı değişiklikleri
korundu. Yeni kaynaklar çalışma ağacından ön kayıt SHA ile sabitlendi.
Windows11 build26200,Python3.12.14,Torch2.10.0+cu128; mevcut .venv/c106r
ve Pixi --locked. CPU inference/thread1/batch1024; GPU eğitimi yok.
CPU Ryzen7 250/RAM24GB önceki cihaz kaydı; bu tur yeniden ölçülmedi.
Env thread kitaplıkları1,CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONUTF8=1.

Robot/TCP4 dosya SHA ve joint/limit sözleşmesi handoff-manifest.json'da;
scaler/checkpoint/config SHA, seed ve roller aynı dosyada. Train/validation
girdileri122 frozen kaynak ve3–9 tanı ön kayıtlarıyla; eski455 teslim dosyası
SHA ile doğrulandı. Eski final raw/query/benchmark açılmadı; yayımlanmış
C1-06 kabul/devir metadata'sı okundu. Yeni bağımsız final oluşturulmadı.

Komutlar depo kökünden:

```powershell
python scripts/c107_command.py 001-preparation -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/prepare_c107_handoff.py
python scripts/c107_command.py 002-preparation -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/prepare_c107_handoff.py
```

001:UTC20:22:27,737–20:22:28,007,exit1; yeni denetleyicide iki fazla
kapanış parantezi SyntaxError; hiçbir ölçüm/registration üretilmeden durdu.
Kaynak düzeltildi, başarısız log korundu.002:UTC20:22:37,639–20:22:50,774,
exit0; Türkiye23:22:37–23:22:50. Modül9,97s,komut13,14s.
Tam argv/env/zaman/stdout-stderr hashleri ../commands/001–002 altında.

## Test ve ham kanıt

PREP-ID:6 checkpoint weights_only yükleme ve kimlik/hash PASS.
PREP-FK:21600 prediction; strict limit, tam-payda A/B ve bağımsız FK
denetimi PASS. Geçerli q için maksimum backend farkı4,622e-16m/
1,038e-15 rotation Frobenius; ön kayıt1e-9/1e-9. Geçersiz q başarısız
sayıldı, limit dışı girdide FK zorlanmadı. Her modelA/B0/3600;
FK_TANH limit dışı0/0/0,RAW1417/1404/1397. Metadata/scaler yanlış eşleşmesi
çıkarsa denetim durur; model girdilerine teacher/split metadata eklenmedi.
PREP-INTEGRITY:122 frozen,455 eski teslim girdisi,3–9 registration PASS.

Ham q/metric: data/generated/C1-07/preparation/{candidate_id}.json;
tam yollar/SHA ve özetler handoff-manifest.json'da. Büyük raw Git dışı.
preparation-audit.json,registration.json,config.json ve komut logları takipte.
Yeni unit test yazılmadı; yalnız doküman/sabit model devir denetimi için
gerçek yükleme ve bağımsız değerlendirme yapıldı. Önceki72 test bu tur
yeniden çalıştırılmadı. Taze ortam T-C06, tüm repo regresyonu, solver
refinement/latency,H1,collision/physical robot ve yeni eğitim NOT_RUN.

## Sonuç ve yorum

Hazırlık denetimi PASS; ürün başarısı PASS değildir. Altı modelin mevcut
ortamda çıkarım zinciri çalışır, doğrudan IK sonucu başarısızdır. Ana model
seçimi tarihsel devamlılıktır; ikincil yerel modelin gerçek hibrit faydası
ölçülmedi. Bağımsız yeni final olmadığı için yeni ürün/final-H2-R kararı
uydurulmadı. C1-06R kapsam durdurması bütün orijinal teslimleri tamamlandı
diye etiketlemez. G1 için temiz ortam/nihai kart/kapanış kararı hâlâ gerekir.

## Sonraki adım

İkinci kullanıcı aşaması tamamlandı; üçüncü aşama C1-06/C1-06R akademik
başarısızlık ve deney süreci raporu için ayrı komut beklenir. O rapor bu
tur yazılmadı. Sonraki teknik paket G1_READINESS.md'deki T-C06/G1 işleri;
Hybrid uygulaması G1 sonrası. STATUS/TRACEABILITY ve görev/roadmap tarihli
eklerle güncellendi, kullanıcı değişiklikleri commit dışında tutuldu.
Bu aşamada push NOT_RUN; yeni eğitim veya faz geçişi yok.

11 Ekim 2026 devam kaydı: ikinci aşamanın son dosya/manifest ve commit
kapsamı kontrolleri tamamlandı. Ölçümler 10 Ekim tarihli002-preparation
koşusuna aittir; yeni eğitim veya ölçüm tekrarı yapılmadı. Kullanıcıya ait
STATUS/TRACEABILITY önekleri korunarak yalnız bu aşamanın ekleri commit'e
alındı. Üçüncü aşama başlatılmadı.
