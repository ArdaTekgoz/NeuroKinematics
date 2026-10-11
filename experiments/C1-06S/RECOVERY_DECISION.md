# C1-06S kurtarma kararı

11 Ekim 2026 · Belge r1 · **PASS / ACCEPTED — ROBOT-SPECIFIC GEOMETRIC IK**.

Kullanıcının doğrulanmış yüksek başarı önceliği doğrultusunda ADR-026 ile
eklenen geometri bileşeni, değişmeyen kinematik ürün hedefini geçti.
Ana küme >=%95 Profil A şartına karşı yeni bağımsız final **10.000/10.000**;
sınır **1.000/1.000**, tekillik **1.000/1.000**. Aynı sayılar daha sıkı
Profil B'de de sağlandı. Yeni finaldeki 12.000 hedef beş defa ölçüldü;
**60.000 ölçüm, 12.000 bağımsız hedef** olarak raporlanır.

Bu kararın kapsamı sabit KR 6 R900 sixx ve hashli tool0 geometrisidir.
Sonuç, örneklenmiş iş yükündeki başarıdır; her olası hedef için matematiksel
tamlık veya fiziksel robot doğruluğu garantisi değildir.

## Kanıt ve kabul matrisi

| Gereksinim | Değişiklik | Test/kanıt | Karar |
|---|---|---|---|
| REQ-C02: doğru robot/kinematik | Foundations değişmeden analitik ters çözüm | 277 F0-01/02/03 regresyonu; 122 frozen hash | PASS |
| REQ-C03: izlenebilir çözüm | Omuz/dirsek/bilek dalları; sınır içi tur seçimi; q_target'sız API | 38 yeni test; ayrı referans FK, köşeler ve tekillikler | PASS |
| REQ-C05 / T-C05S | Önce validation, sonra kod/config freeze ve yeni final | [validation](validation.json), [final](final1.json), [60.000 satır audit](final1-audit.json) | PASS |
| REQ-C06 / T-C06S | Kaynak kimlikleri, bileşen kartı ve tekrar | [ayrı kaynak kopyasında 12.000 birebir replay](clean-replay.json) | PASS; mevcut kilitli ortam |
| Kinematik ürün eşiği | Aynı A <=2 mm/1°, B <=1 mm/0,5°, katı limit | Final A/B 12.000/12.000; yanlış kabul 0 | PASS, yeni geometri bileşeni |
| Eski doğrudan neural IK | Önceki ağırlıklar/eğitim değişmedi | Tarihsel C1-06 ve C1-06R | NO_GO / NOT_MET korunur |
| Eski bilimsel H2 hipotezi | Yeniden değerlendirilmedi | Eski C1-06 | REJECTED korunur |
| Hybrid fazına giriş | Mevcut G1 PASS + yeni doğrulanmış geometri adayı | [HANDOFF.json](HANDOFF.json) | READY_FOR_H2_01 |

Mevcut validation kümesi de **3.600/3.600 A/B**; bunun **351** satırında
eski wide öğretmen etiketi yoktu. Bu yeni çözüm, modele hedef eklemleri
vermeden üretilmiştir. Eski final raw açılmadı. Neural ağ yeniden eğitilmedi;
bu geometri başarısı neural model başarısı diye etiketlenmez.

## Süre ve sınırlar

CPU batch1, tek thread, beş geçiş; aday üretimi, bağımsız FK kabulü ve
dal seçimi dahil bileşen P50 **0,3685 ms**, P95 **0,5029 ms**,
P99 **0,6800 ms**. En uzun ölçüm **103,1543 ms**; 60.000 ölçümün
59.999'u 10 ms ve 50 ms içinde. Aykırı ölçüm çıkarılmadı; nedeni bu
çalışmada saptanmadı. Bu bir iptal/deadline uygulaması veya adil H1
baseline karşılaştırması değildir. Süre garantisi çıkarılamaz.

Yeni final main başarı oranı %100; bağımsız hedefler için Wilson %95
alt sınırı %99,9616. Validation'daki local/wide aynı kökü paylaşır;
oradaki tek-satır Wilson çıktıları bağımsız hedef güven aralığı olarak
kullanılmaz. Başarı oranları bu dağılımlar ve bu robot için geçerlidir.

Taze bağımlılık kurulumu, yeni Linux çalıştırması, H1/ONNX, collision ve
fiziksel robot NOT_RUN. Clean replay ayrı kaynak ağacında, mevcut kilitli
runtime ile yapılmıştır. Kaynak testi kusuru bulunursa yeni sürüm ve
ayrı final gerekir. Mevcut araştırma kayıtları korunur.

## H2'ye somut devir

G1 kabulü zaten vardır; yeni C1-06S bileşeni başarı sağlayan ek girdidir.
Sonraki görev H2-01: doğru kabul/hata/deadline sözleşmesi. Sonra H2-02'de
geometrik adayın servis akışına alınması ADR ile tasarlanır; ilk rapordaki
yalnız neural+refinement akışına sessizce eklenmez. H1, öğrenilmiş politikanın
aynı güçlü tabana ek katkısını ölçer. H2-01 bu görev kapsamında başlatılmadı.

[Neden incelemesi](ROOT_CAUSE_AND_RECOVERY.md), [bileşen kartı](COMPONENT_CARD.md),
[çalışma kaydı](RUN_REPORT.md).
