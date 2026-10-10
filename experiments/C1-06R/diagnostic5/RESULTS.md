# C1-06R tanı5 — Geometri, öğrenme amacı ve veri tasarımı

10 Ekim 2026 · Tanı ve eşli takip tamamlandı · Teknik denetim PASS
Ürün hedefi NOT_MET; C1-06R açık, C1-07 başlamadı.

## Sonuç

Bu tur Foundations kinematik hesabında yeni bir kusur bulmadı. Başarısızlık
tekillik, local quaternion süreksizliği veya durmuş gradyanla açıklanmadı.
Core'da Q kaybının pose hassasiyetini zayıf temsil ettiği ölçüldü; fakat
yalnız amacı değiştiren eşli deney de validation başarısı sağlamadı.
Kök neden tek bileşene indirgenmedi. Öncelikli araştırma konusu artık
verinin yerel öğrenme kapsamı ile modelin geometriyi temsil etme biçimidir.
Bu bir hipotezdir; veri yetersizliği veya mimari kusuru kanıtlanmış değildir.

## Sabit modellerin geometri tanısı

Aynı2048 local train satırı; tanı4 referans ve genişlik512 checkpoint'leri.
Bu bölümde optimizer adımı ve yeni validation çıkarımı yoktur. Kaynaklar
[ADR-019](../../../docs/adr/ADR-019-c106r-train-geometry-diagnosis.md) ve config
ile ölçümden önce kaydedildi; iki modelin ağırlıkları değişmedi.

| Kontrol | Ölçüm | Yorum |
|---|---|---|
| Bağımsız/reference FK,4096 current/teacher | En büyük konum farkı4,69e-16m; dönüş farkı1,09e-15 | 1e-9 sınırı PASS |
| Bağımsız/reference Jacobian,4096 | En büyük normalize fark3,70e-16 | F0-03 1e-5 sınırı PASS |
| Merkezi fark,32 iç teacher satırı | En büyük normalize fark1,84e-10 | Aynı sınır PASS |
| Teacher FK→hedef2048 | Konum farkı0; dönüş farkı1,06e-15 | Etiketler tutarlı |
| Geometrisi en elverişli512 satır | Condition5,59–9,91; referans A0, geniş A1 | Başarısızlık tekilliklerle sınırlı değil |
| Göreli yönelim | En büyük19,17°; quaternion w en küçük0,986 | Local küme π süreksizlik sınırına yaklaşmıyor |
| Aktivasyon/gradyan | Sonlu; bütün katmanlarda sıfırdan farklı gradyan | Toplu NaN/ölü ağ açıklaması desteklenmedi |

Jacobian base eksenlerinde TCP noktasında [lineer;açısal] biçimindedir.
Condition için yalnız konum bloğu0,9015m characteristic length'e bölünür.
Metre ve radyan kontrolsüz biçimde aynı condition hesabına sokulmaz.
Limit içi tahminlerde Jacobian eşliği de doğrulandı; limit dışı21/19 satır
geometri korelasyonlarından ayrı tutulur ve başarı paydasında başarısız kalır.

Geniş modelde J_teacher·dq ile gerçek hata korelasyonu konum0,999795,
yönelim0,999897. Taylor artık medyanı0,212mm/0,041°. Böylece gözlenen pose
hatasının gerçek eklem hatalarından nasıl doğduğu açıklanabiliyor. Bu işlem
çıkarıma eklenmiş bir IK düzeltmesi değildir. Konum lineer katkıları daha
çok joint1–3; yönelim katkıları daha çok joint4–6 tarafında. Katkılar vektör
olarak birbirini söndürebileceği için büyüklükleri toplamsal neden sayılmaz.

## Kayıp ve gradyan

Geniş modelde Q ile Profil A aşım şiddeti Pearson0,304; referansta0,257.
Şiddet=max(konum/2mm,yönelim/1°); yalnız limit içi satırlar için raporlanır.
Küçük Q, küçük TCP hatasıyla aynı şey değildir. Limit aralığı normalizasyonu
aynı radyan hatasını joint6'da joint2'ye göre yalnız0,113 ağırlıkla cezalandırır.
Bu kayıtlı tasarım tercihidir; kodlama kusuru değildir.

| Geniş model, full train | Q | P | R |
|---|---:|---:|---:|
| Ortalama kayıp | 0,0001741 | 0,0002165 | 0,0008585 |
| Parametre gradyanı L2 | 6,91e-5 | 8,93e-3 | 1,46e-2 |

Burada P=||dp/0,9015||², R=||Rpred−Rtarget||F²/8; henüz ağırlık yok.
Q–P/Q–R parametre cosine0,150/0,110; 16 batch'te de pozitiftir. İki
modelin2048'er çıktı gradyanında Q–P ve Q–R negatif cosine sayısı0.
P–R çıktı gradyanlarında554/575 çatışma vardır. Eski raw-model Q–FK çatışması
bu göreli modellerin genel açıklaması olamaz. Norm farkı tek başına en iyi
kayıp ağırlığını veya gradyan kaybolmasını kanıtlamaz.

## Bulgudan sonra ön kayıtlı amaç deneyi

[ADR-020](../../../docs/adr/ADR-020-c106r-profile-scaled-pose-followup.md)
ve [config](loss-followup/config.json) tanıdan sonra, eğitimden önce kaydedildi.
Her iki arm aynı geniş model ağırlığıyla başlar; aynı reset AdamW/cosine,
5000 full-batch adım, seed ve veri. Tek değişiklik Q yerine Profil A ölçekli
pose amacıdır: ||dp||²/(2mm)² + ||dR||F²/[8sin²(1°/2)]. Limit geçerliliği
kayıptan bağımsız olarak aynen uygulanır; clip/solver/ceza eklenmedi.

| Son model | Train A /2048 | Train B /2048 | Train medyan mm/° | Local validation medyan mm/° | Validation A/B /3600 |
|---|---:|---:|---:|---:|---:|
| Başlangıç geniş model, tarihsel | 3 | 0 | 10,94 / 2,89 | 19,65 / 5,55 | 0 / 0 |
| Q'ya5000 adım devam | 57 | 8 | 6,96 / 1,70 | 23,61 / 6,74 | 0 / 0 |
| POSE_A5000 adım | 33 | 1 | 5,96 / 2,39 | 16,81 / 5,47 | 0 / 0 |

Medyanlar burada tam paydadır, invalid +∞ politikası aynen korunur.
Q'da train invalid15, local validation113; POSE_A'da16/95. Her modelde
main validation A/B0/3000 ve wide A/B0/1800. Toplam10.000 update; kampanya
95,13s. Süre eşliği yoktur: Q hücresi20,10s, POSE_A67,65s.

POSE_A konum medyanını iyileştirse de ortak A başarısında Q kontrolünü
geçemedi; validation başarı sıfır. Q'nun eğitim iyileşmesiyle validation
hatasının büyümesi aşırı uyuma işaret ediyor. Kayıp tanımı tek başına bu
bütçede çözüm olmadı; bütün fizik kayıpları elenmiş değildir. Sonuçlara göre
bütçe uzatılmadı. Bu modeller tek seed; bağımsız genelleme kanıtı yoktur.

## Core/Foundations kapsam incelemesi

F0-04 global LHS kapsamını, C1-02 ise her kök için bir local ve bir wide
sorguyu tanımlar. G0 bu veriyle bir ağın2mm/1° öğreneceğini kabul etmemiştir.
7000 main/local train satırında q_current üretimi seed'den birebir tekrarlandı.
Bu kesitte hatalı provenance bulunmadı. Buna karşılık her kökte yalnız bir
local örnek olduğu doğrulandı. 2048 tanı kökünün7000 main train kökü içindeki
en yakın başka kök uzaklığı, local düzeltme büyüklüğünün medyanda6,49 katı
(limit aralığıyla normalize joint L2); bu oran en az1,85. Veri global kapsama
sağlıyor, ancak yoğun yerel yön örnekleri sağladığı varsayılamaz.

Girdi std'leri de farklı: normalize konum0,083–0,088, quaternion xyz0,039–0,045,
q_current0,279–0,289; quaternion w0,00195. Ön işlem/yapısal temsil araştırması
gerekçeli, fakat bu ölçüm tek başına nedensel kanıt değildir. Veri yoğunluğu,
fiziksel anlamlı girdi ölçeklemesi ve periyodik eklem temsili ayrı kontrol
gerektirir; hepsi birden değiştirilerek başarının nedeni belirsizleştirilmeyecek.

[Faz ve literatür incelemesi](PHASE_AND_LITERATURE_REVIEW.md) beş birincil
kaynağı, proje bulgularıyla nerede örtüştüklerini ve sınırlarını kaydeder.
[Yeniden plan](REPLAN.md) yeni veri sürümüyle yerel→global öğrenilebilirliği
ayıran takip yönünü tanımlar. Bu sonraki veri deneyi NOT_RUN.

## Kanıt ve karar

488 test PASS; ilk toplu çağrının6 import collection hatası ayrı loglarda
korundu ve testler izole süreçlerde çalıştırıldı. [Audit](audit.json):4096
tanı satırı,11.296 takip reload satırı,7000 train provenance ve frozen122 PASS.
Kaynak hashleri/ham yolları results.json, sampling-review.json ve checkpoint
kayıtlarında. Büyük satır dosyaları/ağırlıklar Git dışında tutulur.

Yeni uzun kampanya verilmedi. Kabul eşikleri, orijinal Foundations ve eski
Core raporları değiştirilmedi. Yeni final NOT_CREATED, eski final raw NOT_READ;
C1-07/G1 açık. Hedefe ulaştığımızı veya tek kök nedeni bulduğumuzu söylemiyoruz.
