# ADR-026 — Core kurtarma: geometriyi çözüm üretiminde kullanmak

11 Ekim 2026 · Belge r1 · Durum: ACCEPTED_FOR_IMPLEMENTATION

## Bağlam ve yetki

Kullanıcı C1'in sıfır doğrudan IK sonucunun araştırılmasını ve gerekirse
yeniden eğitim/mimari değişikliğiyle NeuroKinematics'in matematiksel olarak
doğrulanmış yüksek başarı hedefine ulaşılmasını istedi. Önceki G1 araştırma
kapanışı ve C1-06/H2 hipotez reddi tarihsel kanıt olarak korunur.

Mevcut MLP deneyleri yalnız başlık/ölçek değişimiyle yeterli hassasiyet
sağlamadı. Önceki kod, FK, teacher ve veri kökeni denetimleri sentetik
verinin yanlış olduğunu göstermedi. Küçük overfit testinden büyük ölçek
genellemeye geçiş yeterince güçlü bir kapıya bağlanmamıştı. Bu ADR yeni
bir MLP mikro-deneyi veya eski sonuçları yeniden adlandırma değildir.

## Karar

C1-06S adlı ayrı kurtarma eki açılır. Sabit KR 6 R900 sixx robotunun
doğrulanmış URDF zincirinden omuz/dirsek/bilek dalları analitik olarak
hesaplanır. Her aday mevcut bağımsız FK ve aynı limit/Profil A/B koşulları
ile doğrulanır. Girdi yalnız hedef poz ve q_current; q_target hiçbir
çözücü arayüzüne girmez. Geçerli dallar arasında normalize sınırlı-eklem
uzaklığıyla seçim yapılır. Sin/cos eşdeğer konfigürasyonlar yalnız gerçek
limit içinde adaydır; q_current'a dairesel uzaklık uygulanmaz.

Bu robotun geometriye dayalı çözücüsü genel bir neural IK yöntemi değildir.
Neural bileşenin gelecekteki rolü dal/arama politikası veya başka robotlarda
öneri üretimi olabilir; katkısı aynı geometrik/sayısal tabana karşı ayrı
H1 deneyinde ölçülmeden üstünlük iddia edilmez. Başarı elde edilirse yeni
bileşen Hybrid'e alternatif seed/teacher olarak devredilir. H2-01/02 servis
sözleşmesi, deadline, fallback ve H1 deneyleri kendi sıralarında kalır.

## Ön kayıt ve kabul

- Foundations girdileri ve eski kaynak/deney kayıtları değişmez.
- Geliştirme: deterministik yeni küçük birim örnekleri, sonra mevcut
  train/validation (eski final açılmaz). Başarısız/etiketsiz satırlar paydada.
- Ayrı final: uygulama ve config hashleri dondurulduktan sonra yeni seed
  ile 10.000 main + 1.000 boundary + 1.000 singularity sorgusu. Her grupta
  %50 local/%50 wide. Yeni finalin yerel farkı ±0,05 rad dahil başlangıç,
  boundary ve singularity eşikleri F0-05 configinden aynen alınır.
  Mevcut C1-02 validation yerel farkı ±0,1 rad olarak ayrıca korunur.
  Üretim ve çift FK denetimi
  modelin başarı sonucundan bağımsızdır. Hiçbir başarısız sorgu çıkarılmaz.
- Main Profil A >=%95; A <=2 mm/1°, B <=1 mm/0,5° ve katı limitler.
  Zor alt kümeler, Profil B ve bütün hata kategorileri ayrıca raporlanır.
- Yanlış kabul sıfır; NaN/boyut/quaternion/limit/TCP ve tekillik testleri.
  Sayısal başarısızlık erişilemezlik kanıtı diye etiketlenmez.
- Final sonrası hata düzeltmesi gerekirse eski final sonucu korunur,
  düzeltilen sürüme yeni final kimliği gerekir.
- Süreler CPU batch1, ön işlem/adayı doğrulama/seçim dahil; beş geçiş.
  Bu bileşen zaman ölçümü H1 kıyası veya servis deadline garantisi değildir.

## Matematiksel yapı

URDF yönelim zinciri Rz(-q1) Ry(q2+q3) Rx(-q4) Ry(q5) Rx(-q6) Ry(pi/2).
TCP'den son 0,080 m x ofseti çıkarılarak bilek merkezi bulunur. Omuz
ofseti 0,025 m, taban yüksekliği 0,400 m, üst kol 0,455 m; ön kolun
etkin uzunluğu hypot(0,420; 0,035) m'dir. Dirsek faz ofseti
atan2(0,035; 0,420) kullanılır. İki omuz × iki dirsek × iki bilek dalı,
limit içi tur eşdeğerleriyle birlikte değerlendirilir. Bilek q5=0
tekilliğinde q4+q6 kısıtı çözülür. Omuz ekseni tekilliğinde q1 için sınırlı
aday araması eksiksizlik garantisi taşımaz; başarısızlık UNRESOLVED kalır.

Geometrik türetim bu deponun hashli URDF'sine özgüdür. Formül doğruluğu
referans Pinocchio ve ayrı URDF-parsed IndependentFK ile sınanacaktır.

## Sonuç sınırı

Başarı, yeni geometri bileşeninin kinematik ürün hedefini karşılamasıdır;
eski ağların başarılı olması, H2 hipotezinin desteklenmesi, H1 hız kazancı,
genel robot desteği veya fiziksel/collision güvenliği anlamına gelmez.
