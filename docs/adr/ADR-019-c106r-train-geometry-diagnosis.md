# ADR-019 — Sabit modellerde train geometri ve öğrenme tanısı

10 Ekim 2026 · ACCEPTED FOR DIAGNOSIS · C1-06R / REQ-C02–05

## Bağlam ve karar

Tanı4 geniş ağ train Q'yu %46 azalttı ama A3/2048; validation0/3600.
Kullanıcı takip tanısını, gerekirse Core/Foundations ve literatür incelemesini
onayladı. Aynı2048 local train satırı ve referans/geniş checkpoint korunur.
Model eğitilmez; inference'a sayısal düzeltici eklenmez. Foundations girdileri
değişmez kalır, bu tur için ayrı kanıt üretilir.

## Ön kayıt

Config ve yeni analiz/test kaynakları çalıştırmadan önce hashlenir.
Teacher ve q_current üzerinde bağımsız FK ve geometrik/Pinocchio Jacobian
karşılaştırılır; ilk32 iç teacher satırında merkezi fark doğrulanır.
Toleranslar F0-03/C1-03 kayıtlarından aynen alınır. Jacobian base ekseninde,
TCP noktasında [lineer;açısal]; characteristic length0.9015m ile normalize
edilir. Metre ve radyan blokları ayrıca raporlanır.

Son tahminlerde dq=q_pred-q_teacher, gerçek pose farkı ve J*dq karşılaştırılır.
Teacher ve limit içi prediction çevresi ayrı; limit dışı q için bağımsız
kinematik API zorlanmaz. Geçersiz satırlar başarı paydasında tutulur.
TrainingFK ideal finite-revolute extension yalnız gradyan analizi içindir.
Train-only condition/limit mesafesi/düzeltme büyüklüğü quartile'ları, Q ile
Profil A eşik aşımı ilişkisi, her eklemin lineer katkı büyüklüğü kaydedilir.
Taylor yaklaşımı geçerli olduğundan emin olmadan katkılar yorumlanmaz.

Q/P/R kayıpları ve parametre/çıktı gradyanları ayrı ölçülür; bütün2048 ve16
eşit batch. SiLU aktivasyonları ve katman gradyanları ölçülür. Küçük değer
otomatik olarak gradyan kaybolması sayılmaz. Göreli quaternion π sınırına
uzaklık ve girdi ölçekleri ölçülür; literatür geneli local veriye körlemesine
uygulanmaz. Seçili modeller/satırlar sonuçlara göre değişmez.

## Tamamlanma ve kapsam sınırı

Her train satırına ait ölçüm, kaynak/ham hash, oracle/FD testleri, en az bir
açıklamanın desteklenmesi veya elenmesi, kalan belirsizlik ve bulguya dayalı
tek sonraki müdahale ön kaydı. Başarısız kontrol takip analizi durdurur.
Sonuç sonrası ek inceleme ayrıca etiketlenir. Yeni validation çıkarımı ve
final yoktur. C1-07, üç-seed uzun eğitim ve ürün kabulü bu tanıdan çıkmaz.
