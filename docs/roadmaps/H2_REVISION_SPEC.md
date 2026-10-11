# H2 r3 ek uygulama şartnamesi

11 Ekim 2026 · ADR-027 · Planlandı; uygulama ve yeni eğitim NOT_STARTED.
Özgün `03_Hybrid_v2_0_r1.md` ile birlikte okunur; eski sayısal eşikleri değiştirmez.

## Rol ayrımı

| Bileşen | Mevcut kanıt | H2 rolü | Henüz kanıtlanmayan |
|---|---|---|---|
| C1-07 eski NN | Dondurulmuş checkpoint/araştırma kapanışı; direct NO_GO | B-H03-old kontrolü, araştırma seed adayı | Refinementte fayda |
| C1-06S analytic-kr6-r1 | Yeni final 12.000/12.000 A/B | Uyumlu KUKA için aday, teacher, B-H05 baseline | Genel robot desteği, deadline, collision, trajectory |
| DLS/LM sayısal | F0/C1 baseline kanıtları | Ortak refinement/fallback motoru | Yeni servis toplam maliyeti |
| H2-N2 yeni NN | Henüz yok | H2-N1 ön kayıtlı yeni veri/mimari ile öğrenilmiş seed | H1/G-N ve direct-neural başarı |

Ürün adayı akış: istek/kimlik → uyumlu geometrik aday → ortak denetim → gerekirse
kalan bütçede numerical → ortak son denetim. Neural-seeded akış ayrı seçilebilir.
G2de varsayılan motor ölçümle seçilir; bu plan neural modeli zorunlu olarak
her sorguya eklemez. Neural inference ek maliyetinin kazanımla karşılanması gerekir.

## Yeni neural kampanya

H2-N1 ve H2-N2 gerçek yeni eğitim işleridir. Amaç başarısız ağı yalnız daha uzun
eğitmek değildir. Veri kök çeşitliliği, eksik wide teacher, branch/tur belirsizliği,
durum/pose şartlandırması ve refinement faydası birlikte ele alınır.

Ön kayıt adayı: K=1 conditioned seed ile basit MLP kontrolü; gerekçeli ikinci kol
K=4 çoklu öneri. Girdi q_current ve hedef pose; sıfır pose farkı/yerel inverse
davranışı ayrı kontrol edilir. Loss/branch eşlemesi, FK loss ölçekleri, sınırlı
joint çıktı temsili ve veri boyutu H2-N1 pilotunda seçilip **yeni ADR ile eğitimden
önce** dondurulur. Bu seçenekler kanıtlanmış en iyi mimari diye sunulmaz.

Öğretmen için mevcut AnalyticIK tek kabul edilen aday verir. Bütün dallar
öğretilecekse ayrı branch API ve validasyon gerekir; private candidate listesi
doğrulanmadan label kaynağı sayılamaz. Sonuç kümesi B profiline bağımsız FK ile
denetlenir. Geometrik öğretmeni olmayan yeni robotta numeric teacher/eksik etiket
politikası korunur. İnference hedef joint etiketini almaz.

Pilot: train hassasiyeti + aynı kökte yeni yön + yeni kökte validation + sınırlı
refinement maliyeti ayrı. 64 örnek ezberleme tam eğitime yeterli GO değildir.
Üç seed; maksimum iki hedefli düzeltme; duvar saati/compute tavanı pilotla kaydedilir.
En iyi checkpoint sadece validationla seçilir. Eski final ve görülmüş C1-06S
finali yeni final değildir; dondurma sonrası yeni bağımsız set üretilir.

## Adil deney kolları

| Kol | Değişen | Sabit kalacak |
|---|---|---|
| B-H01 | q_current seed | Sayısal motor, residual/J, tolerans, iterasyon/bütçe |
| B-H02 | Merkez seed | Aynı koşullar |
| B-H03-old/new | Eski/yeni neural seed | Aynı motor; tüm NN/ön işlem maliyeti dahil |
| B-H04 | Klasik restart | Aynı toplam bütçe, dondurulmuş restart politikası |
| B-H05 | Geometrik servis motoru | Aynı istek/robot/kabul profili ve ölçüm zarfı |

B-H05 ürün karşılaştırmasıdır; H1in kanıt kolu değildir. Çoklu aday kolu açılırsa
her adayın inference/denetim/refinement maliyeti toplamda ve klasik çoklu seed
kontrolüyle ölçülür. Geometry shortcut H1 koluna gizlice eklenmez.

H1: genel ana kümede P95 ≥%20 azalma, başarı kaybı ≤1 yüzde puanı; %95 CI
süre oranı üstü <1 ve başarı farkı altı ≥−1 yüzde puanı. Profil A/B ve 10/50 ms
ayrı; 10.000 ana + 1.000 sınır + 1.000 tekillik, üç seed, beş timing pass,
batch 1/aynı CPU-thread. Başarısız/geç örnekler ve cold start ayrıca raporlanır.

## Çifte kabul kapısı

**G2 servis:** doğru kabul/hata, gerçek toplam bütçe, rollout/collision, ONNX
eşliği ve yeni temiz kurulum. **G-N neural katkı:** bağımsız H1 veya sonradan
ayrı ön kayıtlanmış öğrenilmiş katkı deneyinin kendi koşulları.

H1 negatifse G2nin özgün negatif araştırma sonucu yolu korunur; G-N açık kalır.
Yeni ağın eğitilmesi tek başına G-N değildir. Direct-neural doğruluk iddiası
Profil A/B ile bağımsız ölçülür; seed+refinement başarısına direct denmez.
Neural vizyonu karşılanmamışken proje bu kısmı bitmiş gösterilmez. Sonraki neural
araştırma dalı için G2 raporunda hata grubu, ölçülebilir katkı ve deney ön kaydı
zorunludur; çoklu robot/dal maliyeti gibi ölçülmemiş fikirler başarı diye aktarılmaz.

## Diğer katmanlara şimdi bırakılacak arayüzler

- H2-01: robot/model/solver capability ve validation scope; generic URDF sistemi
  H2de yeniden yazılmaz, MRda kurulur.
- H2-04: ortak state/edge collision API, frame/geometry revision; MP tekrar etmez.
- H2-04 rollout önceki kabul edilmiş q ile; TR generatorın trajectory PASS'i değildir.
- H2-05: yalnız neural tensor hattı ONNX; geometry/host doğrulama ayrı pakette.
- H2-06: temiz servis replay, hashli devir, research-only motor etiketi ve G-N kararı.
