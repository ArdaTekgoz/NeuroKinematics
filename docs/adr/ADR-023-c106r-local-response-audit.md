# ADR-023 — Sabit modellerin sıfır düzeltme ve yerel türev tanısı

10 Ekim 2026 · Kabul edildi · C1-06R tanı8; optimizer/update yok

## Sabit kapsam

Tanı7'nin64/512 köklü RAW/LOCAL_Z dört terminal checkpoint'i değişmeden
incelenir. Model/normalizasyon/veri/kaynak hashleri ölçümden önce kaydedilir.
Eski final okunmaz; yeni final üretilmez; bu bir validation kampanyası değildir.

Sıfır hedef farkı iki anchor'da ölçülür: orijinal eğitim/validation satırının
q_target kökü ve q_current'ı. Her anchor c için hedef pose FK(c), current c.
Train her modelin64/512 kökü; validation tüm1800 local kök. Aynı kimlikler
iki anchor tipinde ayrıca raporlanır. q_hat-c ve sıkı A/B ölçülür.
Üretici bağımsız FK kullanır; referans/bağımsız kinematik çapraz kontrol edilir.

## Küçük yönler ve ölçümün anlamı

Dört modelin ortak ilk64 train kökü ve main local validation içinden,
hem q_current hem q_target limit marjı>0.002rad olan sıralı ilk32 kök.
Her iki anchor tipinde hedef FK(c±h e_j), sabit current c, altı eklem yönü;
h=1e-3,1e-4,1e-5rad. Tam vektörler limit içinde; clipping yok.
Her model128 anchor,4608 küçük-hedef sorgusu. Seçim sıralıdır/rastgele değildir.

K=d q_hat / d q_target. Aynı dalı izleyen yerel inverse için K≈I beklenir.
Farklı geçerli dalda K=I zorunlu değildir. Bu nedenle asıl task-space
karşılaştırması D J(q_hat_zero) K ile D J(c) arasındadır; D'nin ilk3
bileşeni1/0.9015, son3 bileşeni1. Frobenius farkı ||D J(c)|| ile bölünür.
Geçersiz q_hat_zero için prediction Jacobian hesaplanmaz; kapsam açık yazılır.
Alternatif olarak anchor J(c) K farkı, sıfır-bias varsayımıyla ayrıca raporlanır.
K-I göreli normu ||I|| ile bölünür. Yeni bir ürün kabul eşiği türetilmez.

Autograd dq_hat/dx, bağımsız Jacobian'dan feature teğetiyle zincirlenir:
relative p türevi J_linear; quaternion w türevi0,xyz türevi0.5 R(c)^T J_angular.
Scaler türevi uygulanır; current bileşenleri hedefe göre sabittir. Float32
production tepkisi üç h'da finite difference ile karşılaştırılır. Float32
küçük h iptal/yuvarlama etkileri model davranışı diye yorumlanmaz.
Sabit ağırlıkların float64 kopyasında h1e-5 FD-autograd kontrolü C1-03
gradient atol1e-5/rtol1e-3 ile sınanır; kopya bir yeni ürün modeli değildir.

Kontroller: ideal aynı-dal K=I task tepkisini verir; hiç hareket etmeyen
K=0'ın göreli task hatası1'dir. Küçük hedefler A'yı hareketsiz de geçebilir;
tek başına küçük-hedef A oranı inverse öğrenildiğini göstermez. Negatif
işaret/frame mutantları ve bağımsız FK feature FD testleri gerekir.

## Çıktı ve karar sınırı

Sıfır bias, K yön/kazanç, condition ve göreli task türev hatası dağılımları,
fp32/64 FD farkları ve ham satır/matrisler saklanır. Kinematik sınırlar
tanı5 sözleşmesinden alınır. A2mm/1°,B1mm/0.5° aynıdır.
Yerel davranış bozuksa geometrik sıfır/teğet yapısı olan direct-IK mimarisi
sonraki ayrı kontrollü iş olabilir; burada düzeltme/yeniden eğitim yapılmaz.
G0,eski H2,ürün NOT_MET ve araştırma kapanışı ayrımı korunur.
