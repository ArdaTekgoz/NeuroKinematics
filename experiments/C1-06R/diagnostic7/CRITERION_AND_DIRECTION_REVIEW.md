# Validation kıstası mı, araştırmanın gidişatı mı?

10 Ekim 2026 · Kaynak sözleşme + bağımsız hesap + tanı7 değerlendirmesi

**Bulguların ağırlığı öğrenme/genelleme yaklaşımında sorun olduğuna işaret
ediyor. Ölçüm hesabında yeni bir hata bulunmadı. Ancak ürün kabul kıstasını
araştırma ilerlemesinin tek göstergesi ve bütün Core kapanışının şartı gibi
okumak da yanlış.** Bu iki düzeyi ayırmalıyız.

## 1. Ölçütün hesabı ve sağlanabilirliği

Profil A = sonlu/limit içi q VE konum≤0,002m VE geodezik açı≤1°.
Profil B =0,001m/0,5°. Sonuç, seçilmiş teacher eklem vektörüne yakınlığa
göre verilmez; başka geçerli IK dalı da başarılı sayılır. Eksik wide teacher
başarısız model tahminini paydadan çıkarmak için gerekçe değildir.

Bu turdaki kanıtlar:

- Metre/derece, AND ve eşiklerin hemen altı/üstü sentetik testleri PASS.
- NaN, limit ihlali ve etiket eksikliği testinde tam payda korunuyor.
-33984 tahmin bağımsız FK + atan2 rotasyon hesabıyla aynı A/B sonucunu veriyor.
- Validation'ın1800 kökü için local teacher'ı aynı hedefli local/wide
  sorgulara eşleyince3600/3600 A ve B sağlanıyor.351 wide teacher eksikliği
  dahil hedefler mevcut robot modeli içinde erişilebilir tanık taşıyor.

Son madde bir IK modelinin%100 başarısı değildir: oracle değerlendirme
etiketine erişir. Yalnızca kıstasın kendi içinde çelişkili olmadığını ve bu
sorgularda ulaşılabilir çözümler bulunduğunu gösterir; mevcut MLP'nin bu
haritayı öğrenebileceğini garanti etmez. Collision/fiziksel kalibrasyon bu
deneyin konusu değildir. Kanıt [audit.json](audit.json).

## 2. Hedefler nereden geliyor?

[Core raporu §6](../../../docs/raporlar/02_Core_v1_0_r1.md),%95'i başlangıç
ürün hedefi olarak tanımlar; olumlu H2 veya%95, bütün deneyleri bilimsel
olarak kapatmanın şartı değildir. [Ortak test protokolü](../../../docs/TEST_PROTOCOL.md)
A/B toleranslarını proje sözleşmesi yapar. Bu, bütün robotik görevler için
evrensel olarak doğru tolerans veya gerçek kullanımın doğrulanmış ihtiyacı
anlamına gelmez. Bu turda2mm/1°'nin bir fiziksel uygulama ihtiyacından
türetildiğini gösteren yeni doğrulama yapılmadı.

[C1-06R planı](../PLAN.md) ise%95'in main paydasında her üç seed için
aranmasını yeni protokol önerisi olarak açıkça ayırır. Bu ayrıntıyı eski
Core raporunun değişmez tarihsel hükmü gibi sunamayız. C1-06R'de hedefe
ulaşmadan ilerlememe yaklaşımı kullanıcının iyileştirme amacı doğrultusunda
alınan araştırma kararıdır; negatif araştırma kapanışının yasak olduğu
anlamına gelmez. C1-07/T-C06 ve diğer G1 yükümlülükleri ayrıca tamamlanmalıdır;
negatif kapanış da otomatik G1 veya ürün kullanımı izni vermez.

Eşikleri bu başarısız sonuçları PASS göstermek için düşürmüyoruz. Gelecekte
amaçlanan kullanım değişirse yeni ürün gereksinimi ayrı sürüm/ADR ile
tanımlanabilir; eski sonuç ve mevcut NOT_MET kararı değişmez.

## 3. Sıfır başarı küçük bir eşik farkı mı?

Hayır.512-kök RAW kontrolün local validation'ında1737/1800 tahmin limit
içinde; yalnız4'ü konum eşiğini,19'u açı eşiğini geçer; iki koşulu birlikte
geçen yok.1714 geçerli satır iki pose koşulunu da kaçırır. Dolayısıyla
sıfırın ana açıklaması sadece limit kontrolü veya AND kuralı değildir.

Ön kayıtlı açıklayıcı duyarlılık tablosu (yeni kabul eşiği değildir):

| Local validation toleransı | RAW512 | LOCAL_Z512 |
|---|---|---|
|1mm /0,5°|0/1800|0/1800|
|2mm /1° — gerçek A|0/1800|0/1800|
|4mm /2°|2/1800|2/1800|
|10mm /5°|271/1800 (%15,06)|132/1800 (%7,33)|
|20mm /10°|1131/1800 (%62,83)|694/1800 (%38,56)|

İki toleransı10 kat gevşek okumak bile yerelde%95'e getirmiyor. Bu tablo
başarısızlığı yeniden etiketlemek için değil, hatanın eşik sınırında
yığılmadığını görmek içindir. Tam payda/limit koşulu bütün satırlarda sabit.

Bununla birlikte0 başarı “hiç öğrenme yok” demek değildir. q_current'ı
hiç değiştirmeyen referansın local validation medyanı47,44mm/7,16°;
RAW512 modeli16,04mm/4,84°'ye indirir. Bu sürekli ilerleme ürün hedefine
yeterli değildir, fakat sadece0 oranına bakıldığında kaybolur.

## 4. Eğitim görevi ile validation görevi uyuşuyor mu?

Ürün benchmarkı tasarım gereği50/50 local/wide'dır. Mevcut tanılar ise
local öğrenilebilirliğini izole etmek için yalnız local eğitilir. Tam
validation'ı raporlamak doğrudur; onu tek tanı ilerleme metriği yapmak
yanlıştır. Wide başarısı0 kalırsa local%100 olsa bile50/50 toplam en fazla
%50 olur. Bu bir benchmark hatası değil, ürün görevi ile tanı kapsamı farkıdır.

Sonradan açıklayıcı girdi kapsamı kontrolü:512-kök train'de görülen yedi
raw relative pose bileşeninin aralıkları dışında en az bir bileşeni bulunan
validation satırı local14/1800, wide1800/1800. Standardize pose normu
medyanı local2,22; wide317,82. LOCAL_Z wide limit ihlali1799/1800.
Bu, local-only scaler'ın wide kullanımına uygun olmadığını gösterir.
Tek tek bileşen aralığı içinde kalmak ortak dağılım kapsamını veya
genellemeyi garanti etmez; local başarısızlığı da ortadan kalkmıyor.

## 5. Gidişatta düzeltmemiz gerekenler

1. Küçük train overfit'ini uzun eğitim yeterlilik kapısı gibi yorumlamıştık.
   Tanı6'da64/64 train fakat aynı kökte yeni yön0/512; artık bu kapı yeterli
   kabul edilemez. Tanı7 train'i iyileştirip probe'u kötüleştirdi; tek başına
   train loss/başarıya göre model veya normalizasyon seçmemeliyiz.
2. Eşit5000 update karşılaştırmaları kontrollü müdahale etkisini gösterir;
   her modelin optimumuna veya yeterli veri kapsamına ulaştığını kanıtlamaz.
   Ardışık tek-seed küçük denemeler ürünün üç-seed kanıtının yerine geçmez.
3. Aynı validation'ın tekrar tekrar kullanılması onu araştırma verisi yapar.
   Nihai başarı iddiası yeni bağımsız final ister; mevcut validation sonucu
   bağımsız test başarısı diye sunulmaz. Eski final bu süreçte okunmadı.
4. Global sparse kök verisini aynı anda hem yerel hassasiyet hem geniş
   başlangıç dal seçimi için yeterli varsaymak doğrulanmış değil. Düz MLP/Q
   regresyonunda yerel türev davranışını da ölçmeliyiz. Sayısal düzeltici
   eklenirse ayrı hybrid sonuç olur, direct IK eşiği geçmiş sayılmaz.

## 6. Literatür hedefimizi doğruluyor mu?

IKFlow, yaklaşık neural çözümleri ve sayısal refinement'ı ayırır; robotlar
arasında ortalama konum hataları0,36–7,72mm ve açı hataları0,15–2,81°
bildirir. Ortalama hata, sorguların%95'inin iki eşiği birlikte sağlaması
değildir. Büyük ağ/veri ve farklı robot sonuçları bizim MLP için garanti
oluşturmaz. [Birincil kaynak §VI-B/VII-A](https://arxiv.org/html/2111.08933v3).

Neural Inverse Kinematics robot deneylerinde başarıyı10cm konum eşiğiyle
ölçer. O yüksek başarı yüzdeleri bizim2mm+1° ortak kriterimizle eşdeğer
değildir. [Birincil kaynak §5.2.3](https://arxiv.org/html/2205.10837v1),
[yayın kaydı](https://proceedings.mlr.press/v162/bensadoun22a.html).
Kaynaklar10 Ekim2026'da yeniden incelendi. Dolayısıyla%95'i kolayca
ulaşılacak genel bir neural IK standardı gibi beklememeliyiz; bu projenin
sıkı bir hedefidir. Ulaşılamaz olduğu da bu kaynaklardan çıkmaz.

## Karar ve sınırlandırılmış sonraki çalışma

Mevcut A/B ürün kıstası korunur. LOCAL_Z müdahalesi genelleme çözümü olarak
desteklenmedi; yeni uzun eğitim önerilmiyor. Sonraki iş rastgele ek bir
hiperparametre denemesi değil, optimizer çalıştırmadan öğrenilmiş yerel
davranışın tanısı olmalı: sıfır pose farkında sıfır düzeltme ve küçük
yönlerde model çıktısının bağımsız Jacobian'ın gerektirdiği değişimle uyumu.
Bu iki kontrol için ayrıca ön kayıt gerekir; burada NOT_RUN.

Yerel davranış yanlışsa geometrik kısıtlı bir direct-IK çıktı yapısı, doğru
olup yeni köklerde bozuluyorsa kök kapsamı ayrı müdahale olarak ele alınır.
Ancak mimari değişikliği yapılmadan yeni ADR ve aynı veri/bütçe kontrolü
gereklidir. Her iki durumda da local yeteneği ve wide çoklu-dal görevi ayrı
raporlanır; ürün değerlendirmesinde50/50 benchmark korunur. Bu tur tek kök
nedeni veya gelecekteki mimarinin başarısını kanıtlamadı.
