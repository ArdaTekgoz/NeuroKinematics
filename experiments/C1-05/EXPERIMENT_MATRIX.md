# C1-05 ön kayıtlı deney matrisi

8 Ekim 2026 · r1 · Bütün yeni koşular şu anda NOT_RUN.

| Deney / config | Karşılaştırma | λq / λp / λR / λlim | Başlık | Karar ve kapı |
|---|---|---|---|---|
| E-C03 / Q | kontrol | 1 / 0 / 0 / 0 | sınırsız mutlak q | RUN onay + domain/pilot kapıları sonrası |
| E-C03 / FK | Q → FK terimleri | 1 / 1 / 1 / 0 | aynı | RUN aynı kapılarla |
| E-C04 / FK_LIMIT | taze eşli FK → FK_LIMIT | 1 / 1 / 1 / 1 | aynı | RUN E-C03 teknik geçerliliği sonrası; iyileşme şart değil |
| E-C05 / FK_TANH | taze eşli FK → FK_TANH | 1 / 1 / 1 / 0 | normalize (tanh(z)+1)/2 | RUN yalnız seçilmiş E-C03 FK seedlerinden herhangi birinde ham limit ihlali >0 ve teknik kapılar PASS; aksi SKIP |
| E-C06 | quaternion → 6D | — | — | SKIP; ilk dalga FK/limit etkisini önceler, ek temsil bütçesi ayrılmadı |
| E-C07 | mutlak q → delta q | — | — | SKIP; ilk dalgada mutlak q sabit; delta faydası iddia edilmez |
| E-C08 | tekillik cezası | — | — | SKIP; önce hedef family ve geçerli çıktıda ölçekli σmin tanısı; ikinci dalga önkoşulu |
| E-C02 / Res-MLP, curriculum | kapasite / schedule | — | — | SKIP; ayrı ADR ve ön kayıt gerekir |

Her ana çift 2026100201, 2026100202, 2026100203 seedleriyle çalışır. Conditioned
13 giriş, 3×256 SiLU, 136710 parametre, aynı AdamW ve C1-04 train-only scaler.
ℓ=0.9015 m, ADR-005; bütün terimler boyutsuz. Ağırlık1, normalize terimleri
ek bir veriyle ayarlanmış çarpan olmadan sınayan ön kayıtlı tercihtir; optimum
olduğu iddia edilmez. Eski 0.1/0.5/2/5 adayları denenmez.

**Arama fırsatı her arm için bir adaydır.** Adaptif tarama sıfırdır. Olası dört
özgün validation config, sekiz tavanının altındadır; kalan slotlar sonuç sonrası
kullanılamaz. Aynı configin üç seed veya yeni eşli comparator tekrarı yeni arama
fırsatı değildir. E-C04 ve koşullu E-C05 için FK yeniden eğitilir; E-C03'ün
tarihsel seçilmiş checkpointiyle bu taze comparator birbirine karıştırılmaz.

Her çift aynı 15204 labeled train satırını, PCG64(seed) epoch sırasını, 1024
etkin batch'i ve son partial batch'i kullanır. En çok 200 epoch/3000 step/arm;
iki q-validation patience sayacı birlikte20'ye ulaşınca beraber erken durur.
Etiketsiz 1596 train satırı envanterde tutulur ama ilk kontrollü karşılaştırmada
FK fit satırlarına eklenmez. Bu, yarı-denetimli ek veriyi ayrı değişken tutar.

Ana bütçe en çok 18 model-seed koşusu, 54000 optimizer step, 36 saat toplam
hard wall cap; süreç4 GiB, model-seed120 dakika. Bu bir tahmin edilen süre veya
ölçüm değildir. Pilot iki arm ×20 update ve en çok30 dakika,4 GiB; ayrı bütçe.
Tavan aşılırsa PARTIAL ve başarısız koşu saklanır; seed/bütçe azaltarak PASS yok.

Her arm'ın checkpointi **en düşük etiketli validation Lq**, eşitlikte ilk
epoch'tur. En iyi FK epoch'u seçime girmez. Ana araştırma metriği bütün3600
validation sorgusundaki Profil A başarı sayısıdır. C1-06 aday sırası configte
dondurulmuştur; üç seedin tamamı raporlanır ve seed2026100201 çıkarım tanığıdır.
FK üstünlüğü çıkmazsa olumsuz sonuç geçerli araştırma bulgusudur; kayıp/graph
kusuru araştırma bulgusu diye kabul edilmez. G1/H2 kararı bu aşamaya ait değildir.
