# Sabit modelde sıfır düzeltme ve yerel tepki tanısı

10 Ekim 2026 · DIAGNOSTIC8_COMPLETE · Eğitim/güncelleme yok

**Model, hedef zaten mevcut TCP pozuyken çoğu sorguda hedeften uzaklaşan
bir düzeltme üretiyor. Küçük hedef değişimlerine türev tepkisi de ideal
yerel IK ilişkisini karşılamıyor.** Bunlar ölçülmüş davranış kusurlarıdır;
tek bir eğitim/kod hatasının kök neden olarak kanıtı değildir.

## Ön kayıt ve kapsam

ADR-023/config: tanı7'nin n64/n512 RAW/LOCAL_Z dört terminal ağırlığı sabit.
Her modelin64/512 train kökü ve1800 validation local kökü iki anchor ile
sınandı: source q_target (root), source q_current (current). Hedef yeniden
FK(anchor) yapıldı. Bu **sıfır hedef farkı tanısıdır**, eski3600 validation
kampanyası değildir. Aşağıdaki oranlar eski validation0/3600'ı değiştirmez.

Türev için modellerin ortak ilk64 train kökünden32 ve main local validation'dan32,
her iki anchor tipinde limit marjı>0,002rad seçildi. Model başına128 anchor,
6 eklem yönünde±h; h1e-3/1e-4/1e-5rad. Sıralı seçim, tek model seed'i;
genel popülasyon veya nihai test sonucu olarak yorumlanmaz.

## 1. Sıfır hedef farkında davranış

Validation köklerinin **current** anchor'larında1800 sentetik sıfır-hareket sorgusu:

| Sabit model | A | B | Medyan konum / açı hatası | Limit dışı |
|---|---|---|---|---|
|64 RAW|2/1800|0/1800|30,67mm /6,98°|73|
|64 LOCAL_Z|0/1800|0/1800|18,20mm /4,10°|100|
|512 RAW|96/1800|7/1800|5,61mm /1,25°|30|
|512 LOCAL_Z|35/1800|4/1800|7,24mm /1,50°|43|

Medyan tam paydalı nearest-rank; invalid+∞. q_current'ı aynen döndüren
tanık her model/popülasyon/anchor grubunda A/B'nin tamamını sağlar.
Dolayısıyla modelin eklediği düzeltme bu sorguların çoğunda hataya yol açıyor.

Train-root sıfır sorgularında A:64 RAW0/64,64 LOCAL_Z0/64,512 RAW44/512,
512 LOCAL_Z10/512. Train-current karşılıkları0/64,0/64,42/512,10/512.
Sorun yalnız görülmemiş köklerde değil. Ancak sıfır farkı örnekleri özel
olarak üretilmiş yeni tanı girdileridir; mevcut eğitim örneklerinin tekrarı
değildir. Eğitimde aynı hedef çevresinde sonlu perturbasyonlar bulunması,
tam sıfır noktasında doğru davranışı garanti etmemiştir.

q_hat=q_current kesin eşliği bütün IK çözümleri için zorunlu değildir;
başka dal aynı pose'u verebilir. Bu nedenle yalnız eklem değişti diye
başarısız sayılmadı: yukarıdaki A/B doğrudan FK ile pose/limit sonucudur.

## 2. Küçük yönlere verilen tepki

K=d q_hat/d q_target. Aynı dalın ideal yerel cevabı K=I. Alternatif doğru
IK dallarını yanlış cezalandırmamak için temel ölçü:

`||D J(q_hat_zero) K - D J(anchor)||F / ||D J(anchor)||F`

D, konum bileşenlerini0,9015m karakteristik uzunlukla ölçekler. İdeal
tepki0; hiç hareket etmeyen K=0 referansı1 verir. Bu açıklayıcı türev
ölçüsüne yeni bir ürün PASS/FAIL eşiği atanmadı. Geçersiz q_hat için
J(q_hat) hesaplanmaz; aşağıdaki kapsam ayrıca verilir.

| Model | Yeni-kök current anchor geçerli kapsam | Göreli task türev hatası medyanı | Daha düşük condition yarısında medyan |
|---|---|---|---|
|64 RAW|32/32|1,339|1,332|
|64 LOCAL_Z|32/32|1,386|1,504|
|512 RAW|31/32|0,604|0,621|
|512 LOCAL_Z|31/32|0,732|0,764|

512 RAW'da eğitim current anchor medyanı0,556; yani bozulma yalnız yeni
köklerden veya en tekil örneklerden gelmiyor. Yeni-kök current grubunun
Jacobian condition medyanı16,65; düşük-condition incelemesi nitel sonucu koruyor.

Sonradan yön ayrımı:512 RAW'ın31 geçerli anchor'ındaki186 görev-uzayı
yönünün13'ünde beklenen yönle skaler çarpım negatiftir. Medyan yön kosinüsü
0,945, görev tepkisinin kolon norm oranı0,925: model bütünüyle tepkisiz değil,
ancak bütün yönlerde doğru yerel ilişkiyi öğrenmiş de değil. K diagonal
medyanı0,504'ü “TCP hareketi yarıya iniyor” diye yorumlamak yanlış olur;
eklemler ve TCP bileşenleri birbirine karışır. Ham matrisler kaydedildi.

## 3. Sayısal gürültü mü?

Hayır, gözlenen büyük sapmaları açıklayacak düzeyde değil. Tüm gruplarda
sabit ağırlıkların float64 kopyasında autograd-zincir ve merkezi fark
C1-03 atol1e-5/rtol1e-3 kontrolünü geçti. Shadow64 FD matris farklarının
gözlenen en büyük Frobenius normu yaklaşık2,46e-7.

512 RAW yeni-kök current grubunda fp32 FD/autograd normalize fark medyanı
h1e-3'te0,000243; h1e-4'te0,00234; h1e-5'te0,0267. Çok küçük h'da
yuvarlama/çıkarma etkisi artıyor; türev yorumunu yalnız o stencil'e dayandırmadık.
Autograd ve daha büyük h kontrolü,0,604 task türev sapmasını destekliyor.
Float64 kopya yeni eğitilmiş ürün modeli değildir; orijinal ağırlıklar değişmedi.

## 4. Küçük-hedef başarı oranı neden yeterli değil?

Üç h ile toplam18432 küçük-hedef sorgusunda, q_current'ı hiç değiştirmeyen
referans zaten A/B18432/18432 sağlar. Adımlar toleransların içinde kalır.
Buna karşın512 RAW'ın seçilmiş32 yeni-kök current anchor'ında her h için
model A12/384, hareketsiz referans384/384'tür. Sıfırdaki model hatası küçük
hedeflerde de baskın kalabilir. Bu yüzden küçük-hareket A oranı, doğru
inverse davranış öğrenildiğini tek başına kanıtlayamaz; türev kontrolü gerekir.

## Kanıt ve sonraki karar

68 test PASS;128 anchor'da reference/independent FK/Jacobian ve128 FD
kontrolü PASS (zero current/teacher aynı olduğu için256 karşılaştırma
satırı128 anchor'ı temsil eder). Audit16704 sıfır sorguyu,18432 küçük
sorguyu ve512 model-anchor türevini tekrar üretti.122 frozen kaynak ve
checkpoint hashleri aynı; optimizer adımı0. Yeni final NOT_CREATED,
eski final NOT_READ. [Audit](audit.json),[çalışma kaydı](RUN_REPORT.md).

Bu bulgular başarısızlığı yalnız validation eşiğine veya global veri
seyrekliğine bağlamayı desteklemiyor. Sıfır ofsetini düzeltmek gerekçeli
bir mimari deneyi olur, fakat türev kusurunu otomatik çözmez.
[Sonraki öneri](NEXT_EXPERIMENT.md) öncelikle sıfır düzeltmeyi yapısal olarak
sağlayan residual başlık; ayrı ön kayıt/eşli eğitim gerektirir ve NOT_RUN.
Mevcut modelin eşik geçmesi veya bu mimarinin başarılı olacağı iddia edilmez.
