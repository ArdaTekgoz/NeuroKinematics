# Deney veya uygulama kaydı

Kimlik: RUN-20261011-C106S  
Durum: COMPLETE / GEOMETRIC_PRODUCT_GATE_PASS / READY_FOR_H2_01  
Görev ve gereksinim: C1-06S; REQ-C02/03/05/06, T-C05S/T-C06S  
Tarih ve sorumlu: 11 Ekim 2026; kullanıcı yetkisiyle Codex

## Soru ve değişiklik

Sıfır neural başarı hedeflerin/ölçütün bozuk olmasından mı kaynaklanıyor;
sabit robot geometrisiyle aynı sıkı koşullarda yüksek başarı elde edilebilir
mi? Kullanıcı gerekirse mimari değişikliğini onayladı. ADR-026 ve protokol
sonuçlardan önce yazıldı. Yeni analytic.py, geçerli dalları URDF geometrisiyle
üretip IndependentFK ile kabul eder. Ağın yeniden eğitilmesi veya eski
H2 hipotezinin olumluya çevrilmesi bu çalışmanın sonucu değildir.

Matematiksel yapı, veri/öğrenme incelemesi ve kapsam [ADR-026](../../docs/adr/ADR-026-geometric-core-recovery.md),
[ROOT_CAUSE_AND_RECOVERY](ROOT_CAUSE_AND_RECOVERY.md), [COMPONENT_CARD](COMPONENT_CARD.md).
Ölçütler değişmedi: A <=2 mm/1°, B <=1 mm/0,5°, sonlu ve limit içi q;
main A >=%95 ürün kapısı. Karar [RECOVERY_DECISION](RECOVERY_DECISION.md).

## Tekrar üretim

Başlangıç HEAD `0f518d24dadd83b4e7856f6cd2eac3bee17d2da2`, branch
`codex/c1-06r`. Başlangıçta kullanıcı STATUS/TRACEABILITY değişiklikleri
ve PDF/DOCX/AUDIT dosyaları vardı; bunlar korundu. Yeni kod commit/push
yapılmadı. Kirli ağaç tam kaydı [final-freeze](final-freeze.json) içindedir.

Windows 11 build26200, Python 3.12.14, NumPy 2.5.3, Pinocchio 4.1.0,
Pixi 0.81.0; mevcut `.venv/c106r` + kilitli Pixi runtime. CPU Ryzen7 250,
RAM24GB önceki doğrulanmış donanım kaydı; bu çalışmada donanım yeniden
envanterlenmedi. CPU BLAS/OMP thread1; batch1. CUDA eğitimi yok.
Robot/TCP dört hash ve 122 eski kaynak/girdi byte hash'i değişmez.
Yeni 23 final kaynağı/girdisi final-freeze'de. Ağırlık/scaler yok.

Unit seed2026101101/02; yeni final PCG64 seed2026101121 ve +1…+4
bağımsız akışları. F0-05'ten aynı ana/sınır/tekillik dağılımı ve başlangıç
kuralları; validation C1-02'nin ±0,1rad local kuralını korur, yeni final
F0-05'in ±0,05rad local ve wide >=0,25 normalized RMS kuralını kullanır.
Final öncesi freeze UTC00:14:30; final UTC00:14:36–00:15:25 (49,21s).
Türkiye +3. Saatler command.json dosyalarında tam UTC olarak saklandı.

Bütün komutlar aşağıdaki wrapper ile çalıştı:

```powershell
python scripts/c106r_command.py ATTEMPT -- pixi run --locked .venv/c106r/Scripts/python.exe COMMAND
```

| ATTEMPT (experiments/C1-06R/commands altında) | COMMAND | Sonuç |
|---|---|---|
| 062-geometric-unit | -m pytest tests/c1_06s -q --junitxml=experiments/C1-06S/unit.xml | 32 PASS/2 FAIL; tekillik yakınında açı ayrışımı kaybı |
| 063-geometric-unit-r2 | -m pytest tests/c1_06s -q --junitxml=experiments/C1-06S/unit-r2.xml | 34 PASS; korelasyon korunarak son bilek açısı düzeltildi |
| 064-geometric-validation | scripts/evaluate_c106s.py validation | 3600/3600 A/B |
| 065-geometric-validation-audit | scripts/evaluate_c106s.py audit-validation | PASS, 3600 bağımsız FK |
| 066-geometric-regression | -m pytest tests/c1_06s tests/f0_01 tests/f0_02 tests/f0_03 -q --junitxml=experiments/C1-06S/regression.xml | 3 collection ERROR; eski test modül adları çakışıyor |
| 067-geometric-isolated-regression | scripts/test_c106s.py | 38 yeni+277 Foundations =315 PASS; ayrı süreçler |
| 068-geometric-freeze | scripts/evaluate_c106s.py freeze | PASS; yeni final henüz üretilmemişti |
| 069-geometric-new-final | scripts/evaluate_c106s.py final | 12000 sorgu×5 geçiş=60000; tüm A/B PASS |
| 070-geometric-final-audit | scripts/evaluate_c106s.py audit-final1 | 60000 FK, tam payda ve input seed replay PASS |
| 071-geometric-clean-source-replay | scripts/reproduce_c106s.py | 12000 q/metrik birebir, ayrı kaynak ağacı |
| 072-geometric-closure | scripts/finalize_c106s.py | Kabul, kaynak/kanıt manifesti ve H2 devir eki |

Wrapper gerçek argv, seçili ortam, başlangıç/bitiş, exit ve log SHA'larını
kaydeder. İlk iki başarısız komut/log/XML korundu. Eşik düşürülmedi.
Pytest record_property/xunit2 uyarıları eski testlerde devam ediyor;
son koşuda fail/error/skip sıfır.

## Test ve ham kanıt

38 yeni test:512 rastgele pose, tüm64 kutu köşesi,12 ayrı sınır, bilek ve
dirsek tekilliği/komşuluğu, omuz ekseni, limit içi turlar, quaternion işareti,
bozuk girişler, yanlış TCP mutantı, uzak hedefin UNRESOLVED olması ve API'de
q_target bulunmaması. F0-01:16, F0-02:102, F0-03:159 regresyon PASS.

[validation.json](validation.json): main3000/boundary300/singularity300,
local1800/wide1800, eksik wide teacher351; hepsi A/B PASS. Sadece sorgu
pozları/q_current çözücüye girer. Eski label veya q_target kullanılmaz.

[final1.json](final1.json): main10000/boundary1000/singularity1000;
local6000/wide6000. A/B12000/12000, invalid0, false_acceptance0.
Hedefler çözücü çalışmadan önce üretildi, reference/independent FK ile
denetlendi, ayrı witness dosyasına kaydedildi. 60.000 ölçümden başarısız
veya yavaş satır çıkarılmadı. [audit](final1-audit.json) yeni veri üretimini
aynı seedlerle birebir tekrarladı, bütün sonuçları ayrı FK ile hesapladı.

Ham kayıtlar: data/generated/C1-06S/validation/{requests.json,results.jsonl};
final1-inputs/{requests.json,witnesses.json}; final1/results.jsonl.
SHA'lar validation/final1-inputs/final1/audit dosyalarındadır. Git dışındadır.
Eski final raw NOT_READ. Yeni final1 geometri içindir; eski H2-R neural
finali hâlâ NOT_CREATED/NOT_EVALUATED, ikisi aynı deney değildir.

Clean replay: git archive başlangıç HEAD + 23 hashli ek/girdi ile ayrı
temp/c106s-clean-source; mevcut runtime, kaynak önceliği bu kopyaya ait.
12000 çıkarım/Pinocchio metriği EXACT. Taze ortam kurulumu NOT_RUN.

## Sonuç ve yorum

Yeni geometri bileşeni ürünün kinematik başarı kapısını geçti. Eski neural
ağlar değişmedi; doğrudan neural IK ve eski H2 kararları değişmedi. Başarıyı
sağlayan değişiklik, robotun çözüm geometrisini üretim aşamasına almaktır.
Sentetik hedeflerin çözülebilirliği bağımsız olarak gösterildi; eski MLP
başarısızlığının veri/kapasite/amaç etkileri tek nedene indirgenmedi.

P95 bileşen süresi0,5029ms, max103,1543ms. 59999/60000 ölçüm10ms içinde;
tek yavaş ölçüm korunur. H1 karşılaştırması, servis deadline garantisi,
collision, fiziksel robot, Linux ve ONNX NOT_RUN. Mevcut robotta gözlenen
%100 her hedefte garanti değildir; omuz ekseni dal taraması sınırlıdır.

## Sonraki adım

C1-06S COMPLETE/PASS; G1 araştırma kabulü korunur, H2-01'e ek girdi hazır.
H2-01 kabul/hata sözleşmesi → H2-02 geometri+sayısal/neural aday yönetimi
→ H1 katkı deneyi. Neural bileşenin basit geometri tabanına ek faydası
ölçülmeden üstünlük iddiası yok. Bu tur yeni neural eğitim/H2 servisi yok.
STATUS/TRACEABILITY ek güncellemesi kullanıcı değişikliklerini korur.
Eski negatif rapor ve LinkedIn taslağı tarihsel kalır; yayın yapılmadı.
