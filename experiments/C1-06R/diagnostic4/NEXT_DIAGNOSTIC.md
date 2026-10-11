# Tanı4 sonrası takip planı

Durum: PROPOSED / NOT_RUN · 10 Ekim 2026 · C1-06R, C1-07 başlamaz

## Yanıtlanacak soru

Genişlik512 ile train Q %46 azalırken train A yalnız3/2048 ve validation
A0/3600. Kalan hata hangi örnek/geometri özelliklerinde yoğunlaşıyor ve
mevcut Q amacı kabul edilen TCP hassasiyetini ne ölçüde temsil ediyor?
Yeni uzun eğitimden önce bu soruya train kanıtıyla yanıt aranacak.

## Sıra ve çıktılar

1. Aynı2048 train örneği ve sabit referans/geniş model checkpoint'leriyle
   eklem hatası ile TCP hata ilişkisini ölç. Doğrulanmış Jacobian sözleşmesini
   oku; sayısal türev kontrolünden sonra teacher/prediction çevresindeki
   yerel duyarlılığı karşılaştır. Birim ve referans çerçevesi karışmayacak.
   Konum/yönelim blokları ayrı analiz edilecek; birimsiz ölçek tanımlanmadan
   birleşik Jacobian condition sayısına anlam yüklenmeyecek.
2. Q kaybı ile gerçek pose eşik aşımını örnek bazında karşılaştır. Kayıp
   ağırlığının yanlışlığını önceden varsayma. Train-only gruplarla eklem
   limit yakınlığı, geometrik hassasiyet ve hedef düzeltme büyüklüğü ayrıştır.
   Veri kökeni/etiket sürekliliği için bulunan somut şüpheli satırları,
   önceki bağımsız FK oracle kanıtıyla birlikte incele.
3. Sabit modellerin train gradyan/aktivasyon dağılımını incele. Önceki
   round1 Q/FK çatışma bulgusunun bu yeni temsil/modelde geçerli olduğu
   varsayılmayacak. Yeni fizik kaybı düşünülürse Q ve FK gradyanları ayrı ölçülür.
4. Bulgulara göre tek sonraki müdahale seçilir ve yeni ADR/config ile ön
   kayıt yapılır. Geniş train üzerinde hedef hassasiyeti gösterilmeden üç
   seed uzun kampanya başlatılmaz. Validation'a göre art arda checkpoint
   veya alt örnek seçilmez; nihai iddia için bağımsız yeni final gerekir.

## Tamamlanma ölçütü

Train satırlarının tamamı için denetlenebilir bir hata/duyarlılık tablosu,
en az bir açıklamanın ölçümle desteklenmesi veya elenmesi, kalan belirsizlik
ve tek müdahaleli deney gerekçesi. Betimsel korelasyon kök neden diye
sunulmaz; donanım/kapasite yetersizliği sırf başarısızlıkla ilan edilmez.
Geometri analizi çözüm sonuna sayısal düzeltici eklemeyecek; hybrid C1-07
ayrı bağımlılık olarak kalır. Mevcut faz kabul eşikleri değişmez.
