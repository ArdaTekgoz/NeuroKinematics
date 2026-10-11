# Deney veya uygulama kaydı

Kimlik: RUN-20261010-C106R-DIAGNOSTIC2

Durum: COMPLETE_DIAGNOSIS / VALIDATION_TARGET_NOT_MET / PRECISION_FIX_VERIFIED

Görev ve gereksinim: C1-06R; REQ-C02–05; eşli tanı ve kapsamlı hata incelemesi

Tarih ve sorumlu: 9–10 Ekim 2026; AI kısa tanı/denetim; uzun eğitim NOT_RUN

## Soru ve değişiklik

Kullanıcı sonraki tanıyı çalıştırmamızı, başarısızsa projeyi kapsamlı
kontrol etmemizi istedi. ADR-015 ve config ile 8 eşli absolute/residual,
local/mixed, 64/512 örnek tanısı sabitlendi. Dört ek kayıtlı aynı-checkpoint
LBFGS tanısı uygulandı. Beklenen 64/64 train küçük tanı geçti ama validation
genellemesi gelmedi. Bunun üzerine robot→veri→öğrenme→ölçüm zinciri
incelendi. Doğru teacher oracle ile endpoint dönüşüm hatası ayrıldı;
ADR-016 kapsamında eski kodu koruyan yeni float64 decoder üretildi.

## Tekrar üretim

Başlangıç HEAD `0d8762c`; `codex/c1-06r`. Kullanıcının STATUS/TRACEABILITY
üst değişiklikleri ve ilgisiz PDF/DOCX/AUDIT dosyaları korunur.
Windows, RTX 5060 Laptop 8 GB / 24 GB RAM; aynı locked Pixi ve
`.venv/c106r` Torch 2.10.0+cu128. Float32 öğrenme, TF32/AMP kapalı,
deterministik ve CPU thread 1. Yeni decoder float64, gradyan korunur.
Robot/TCP, train/validation veri kimliği ve round1 kodu değişmez.

`config.json`, `registration.json`, `refinement-config.json` ve
`refinement/registration.json` deney öncesi config/code/source hashlerini
kaydeder. Tanı seed'i 2026100901; 8 × 5000 = 40.000 AdamW güncellemesi.
Her 512 takip koşusunda 500 LBFGS iterasyonu; closure sayıları 526, 523,
515, 531. Bütün seçimler son adım; validation ile seed/epoch eleme yok.
Ağırlık ve satır ham çıktıları data/generated/C1-06R/diagnostic2 altında
Git dışı; JSON sonuç dosyaları yol ve SHA içerir.

Komut girişleri `../commands/<kimlik>/command.json` içinde argv/cwd/env,
UTC başlangıç/bitiş, exit code ve stdout/stderr hashleriyle kayıtlıdır:

| Kimlik | Komutun işlemi | Sonuç |
|---|---|---|
| 025-diagnostic2-tests | pytest tests/c1_06r/test_diagnostic2.py | 3 PASS |
| 026-diagnostic2-matrix | python -m neurokinematics.neural.c106r_diagnostic2 | 8 kısa koşu tamamlandı |
| 027-diagnostic2-runtime-review | run_c106r_diagnostics.py runtime, yeni project-review çıktısı | PASS |
| 028-diagnostic2-data-review | run_c106r_diagnostics.py audit-data, yeni çıktı | PASS |
| 029-diagnostic2-refinement | refine_c106r_diagnostic2.py | 4 kısa takip tamamlandı |
| 030-review-* | f0_01/f0_02/f0_03/c1_03/c1_04/c1_05 ayrı pytest süreçleri | 435 PASS |
| 031-pipeline-oracle-review | review_c106r_pipeline.py | Oracle/bağımsız FK/alt küme analizi |
| 032-review-c106r | pytest tests/c1_06r | 33 PASS |
| 033-precision-fix-tests | pytest tests/c1_06r/test_precision.py | 6 PASS |
| 034-precision-fix-impact | audit_c106r_precision.py | Yeni decoder ve 20 model etki denetimi PASS |

Her giriş `python scripts/c106r_command.py <kimlik> -- pixi run --locked
.venv/c106r/Scripts/python.exe ...` kaydedicisiyle çalıştı. Yeniden koşu
önceki sonuçları ezmez; yeni deney kimliği/çıktı yolu gerekir.
026 UTC 9 Ekim 20:18:34–20:20:47 (133 saniye süreç; hücrelerin toplamı
125,47 saniye). Son etki ölçümü UTC 21:34:26–21:34:47; Türkiye'de 10 Ekim.
Kullanıcının devam mesajı öncesinde tamamlanan koşular yeniden eğitilmedi.

## Test ve ham kanıt

Yeni tanıda 64 hücrelerin her biri A64/64; 512 hücreler A0/8/487/419.
LBFGS sonrası A0/24/508/491. Sekiz hücre ve dört takip validation A0/3600.
Kapsamlı regresyon toplamı 474 PASS, 0 fail/skip; 025'teki üç test 032
içinde tekrar yer aldığından iki kez sayılmadı. XML dosyaları project-review
altında. C1-03 NaN mutant uyarısı korunur, ürün çalıştırma hatası değildir.

20.400 veri satırı provenance/normalizasyon ve 18.453 teacher label A/B
doğrulandı. IndependentFK/Pinocchio azami fark train yaklaşık 6,67e-16 m,
9,52e-16 matris normu; validation 4,50e-16 m, 9,24e-16. CPU/GPU ve
FP32/64 kinematik/türev raporu runtime.json. Yanlış sıra, ölçek, label,
NaN ve payda negatifleri PASS.

Eski float32 decoder doğru teacher cevaplarından 100 train ve 19
validation satırını limit dışına taşıyor. Yeni decoder öğretmen train
15.204/15.204 ve validation3249/3249 A/B sağlar; katı limit ihlali sıfır.
6 yeni CPU/GPU/gradcheck testi PASS. Eski 12 round1 model ve yeni 8 tanı
modelinde düzeltme sonrası validation yine0/3600; bu kusur temel başarısızlık
açıklaması değil. Eski fizik kaybı/round1 hashleri değiştirilmedi.

## Sonuç ve yorum

[RESULTS](RESULTS.md) kanıtlanan kusurları, öğrenme davranışını, süreç
hatasını ve nedensel olarak kanıtlanmamış açıklamaları ayırır. Teacher
dal farklılığı ve orta nokta başarısızlığı etiket yanlışlığı değildir;
q_current'lar farklıdır. Ürün başarısı, H2-R desteği veya Hybrid hazır
oluşu iddia edilmez. Kapsamlı inceleme neural zincire odaklıdır; harici
Docker solver/final kampanyaları ve C1-02 sealed raw tam kabul tekrar
çalıştırılmadı. Yeni final NOT_CREATED, eski final raw NOT_READ.

## Sonraki adım

Aynı 13 boyutta q_current FK'sine göre göreli pose ön işleme tanısı,
512 ve daha geniş ölçek geçişi; sonra kanıt varsa üç-seed uzun paket.
Bu yeni temsil deneyi henüz NOT_RUN. Decoder ve plain-string metadata
sonraki ayrı sürüme girecek; production resume testi teslim şartı.
Mevcut eski ağırlık/raporlar korunur; C1-07/G1 açık kalır.
