# Deney veya uygulama kaydı

Kimlik: RUN-20261010-C106R-DIAGNOSTIC5

Durum: COMPLETE_DIAGNOSIS_AND_PAIRED_FOLLOWUP / AUDIT PASS / TARGET_NOT_MET

Görev ve gereksinim: C1-06R; REQ-C02–05; geometri/öğrenme ve faz incelemesi

Tarih ve sorumlu: 10 Ekim2026; kullanıcı tanıyı ve gerekirse kapsamlı
Core/Foundations/web incelemesini onayladı; AI uygulama/test/analizi yürüttü.

## Soru ve değişiklik

Tanı4 sonrasında hangi geometri/öğrenme etkeni hassasiyeti engelliyor?
ADR-019/config ile aynı2048 local train satırında referans/geniş modelin
FK/Jacobian, Taylor, gradyan, aktivasyon ve girdi ölçekleri ölçüldü.
Sabit modeller bu analizde değiştirilmedi. Sonuca göre ayrıca ön kayıtlı
ADR-020 iki amaç arm'ı çalıştırıldı: aynı geniş modelden Q ve POSE_A5000'er
adım. Önceden beri planlanmış gibi gösterilmedi. Sonuç sonrası train örnekleme
incelemesi ve birincil kaynak araştırması eklendi. Eski kaynak/kanıt değişmedi.

## Tekrar üretim

Başlangıç HEAD72f1765, branch codex/c1-06r; yeni kod ve belgeler çalışırken
kirli ağaç. Kullanıcı STATUS/TRACEABILITY değişiklikleri ve ilgisiz
PDF/DOCX/AUDIT çıktıları korunur. Yazılım hedefi v1.0.0; görev belgesi r6.
Windows11, AMD Ryzen7 250, RTX5060 Laptop8151MiB; kullanıcı beyanı24GB RAM.
Locked Pixi/.venv/c106r; Python3.12.14, Torch2.10.0+cu128. CPU thread1;
CUBLAS workspace4096:8, TF32/AMP kapalı, deterministik seed2026100901.

Analiz model ağırlıkları tanı4 reference/capacity;13 göreli girdi, residual
head,256/512 üç SiLU katmanı;136710/535558 parametre. Yeni takip iki arm'da
aynı capacity ağırlığı, aynı reset AdamWlr.001/wd.01, cosine eta1e-6,
5000 full-batch step. Float32 ağ; endpoint-exact float64 decoder ve fizik
amacında float64 TrainingFK. POSE_A normu2mm/1° ölçeklidir; bunun kabul
testinin yerine geçmediği ADR-020'de belirtilmiştir. Son checkpoint seçilir.

Robot/TCP/veri/kilit kaynakları historical training-freeze122; yeni kaynaklar
diagnostic5/registration.json ve loss-followup/registration.json içinde.
Örnekleme incelemesi source hashlerini sampling-review.json'da kaydeder.
Büyük ağırlık/ham satırlar data/generated/C1-06R/diagnostic5 altında
LOCAL_ONLY, her çıktı için yol/SHA kayıtlı. Önceki kayıtlı kaynaklar korunur.

```powershell
python scripts/c106r_command.py 043-geometry-foundations-core-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/f0_01 tests/f0_02 tests/f0_03 tests/c1_03 tests/c1_04 tests/c1_05 tests/c1_06r -q --junitxml=experiments/C1-06R/diagnostic5/tests.xml
python scripts/c106r_command.py 044-geometry-isolated-regression -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/check_c106r_diagnostic5_regression.py
python scripts/c106r_command.py 045-train-geometry-learning-diagnosis -- pixi run --locked .venv/c106r/Scripts/python.exe -m neurokinematics.neural.c106r_diagnostic5
python scripts/c106r_command.py 046-profile-pose-objective-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/c1_06r/test_pose_followup.py -q --junitxml=experiments/C1-06R/diagnostic5/loss-followup/tests.xml
python scripts/c106r_command.py 047-profile-pose-objective-pair -- pixi run --locked .venv/c106r/Scripts/python.exe -m neurokinematics.neural.c106r_pose_followup
python scripts/c106r_command.py 048-train-sampling-review -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/review_c106r_train_sampling.py
python scripts/c106r_command.py 049-geometry-and-followup-audit -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_c106r_diagnostic5.py
```

Exact argv/env/start/end/exit/log SHA ../commands/043–049 altında. UTC10Ekim:
testler04406:23:45–06:24:25 (40,35s); tanı04506:24:34–06:24:48 (13,57s);
takip04706:27:42–06:29:17 (95,13s); sampling04806:30:08–06:30:13;
audit04906:31:51–06:31:56. Türkiye saati UTC+3. Bunlar insan emeği değildir.
Kimlikler immutable; aynı output'u ezmek yerine yeni revizyon gerekir.

## Test ve ham kanıt

- 043 FAIL_COLLECTION: aynı test modül adları/conftest ithalleri toplu
  süreçte çakıştı;6 collection error. tests.xml ve log korundu. Eski test
  kodu değiştirilmeden her klasör ayrı süreçte çalıştırıldı.
- 044 PASS484: F0-01=16, F0-02=102, F0-03=159, C1-03=110,
  C1-04=12, C1-05=36, C1-06R=49. C1-03 NaN mutantından bir determinant
  uyarısı logda. Fail/skip0. Yeni tanı testleri3 bu sayıya dahil.
- 045 PASS:4096 current/teacher FK+Jacobian,32FD,2048 teacher/target;
  iki modelde4096 satır inceleme, ayrı train replay ve değişmeyen ağırlıklar.
- 046 PASS4: sıfır pose kaybı,2mm sınırı,1° sınırı, iki sınırın toplamı.
  Toplam bu tur başarılı test sayısı488; yeni test sayısı7.
- 047 COMPLETE: eşli10.000 update; train/validation11.296 satır güvenli
  checkpoint reload EXACT_MATCH. Q train A57/B8, POSE_A A33/B1; her iki
  validation A/B0/3600. Full payda/invalid politikası korunur.
- 048 PASS: yalnız train okunarak7000 main/local provenance tekrarlandı;
 2048 kökte nearest-other-root ölçümü, ham satırlar hashli. Model seçimi yok.
- 049 PASS: registered/ham hashler,122 freeze,488 JUnit sonucu, aynı
  başlangıç ağırlığı/örnek listesi, payda/eşikler ve sayılar denetlendi.
- F0-04/05/06 tam üretim/benchmark kampanyası, Docker/temiz ortam kurulumu,
  üç-seed yeni uzun eğitim ve fiziksel robot deneyi bu tur NOT_RUN.
  Eski final raw NOT_READ, yeni final NOT_CREATED.

## Sonuç ve yorum

[RESULTS](RESULTS.md) ayrıntılı sayıları verir. Bu örneklerde yeni FK/
Jacobian/etiket kusuru bulunmadı; local quaternion π sınırından uzak,
düşük condition grubunda da başarı düşük, gradyan/aktivasyonlar sonlu.
Q/TCP hata ilişkisi zayıf; fakat amaç takibi validation başarısını artırmadı.
POSE_A local konum medyanını16,81mm'ye indirir; A0/3600. Q train iyileşirken
local validation23,61mm/6,74°'ye kötüleşir. Kayıp seçimi tek başına yeterli
olmadı. “Foundations hatasızdır” veya “veri kesin yetersizdir” denmiyor.

7000 global main kökte yalnız bir local örnek; en yakın diğer kök/local
düzeltme uzaklığı oranı medyan6,49. G0 global örnekleme/hesap doğruluğunu
kabul etmiş, hassas inverse öğrenilebilirliği kabul etmemiştir. Literatür
incelemesi çoklu çözüm, temsil sürekliliği, gradyan dengesi, örnek ölçeği
ve uyumsuz başarı eşiklerini ayırır; yayın sonuçları proje sonucu sayılmaz.

## Sonraki adım

[REPLAN](REPLAN.md): C1-02R sürümlü directional local veri kontrolü;
aynı-kök yeni yön tanısı ile farklı-kök validation'ı ayrı ölçmek. Sonuç
gerektirirse tek temsil müdahalesi. Bu yeni veri deneyi NOT_RUN.
Eski kaynaklar/devir değişmez; eşikler düşmez. C1-07/G1 ve ürün hedefi
açık, yeni uzun kampanya henüz hazır değil.
