# C1-06 Aşama 2 açılış öncesi kontrol

8 Ekim 2026. Kullanıcının “Onaylıyorum” mesajı approval.json içinde Stage1
commit ve SHA256SUMS hashine bağlandı. Stage1 girdileri 322/322 PASS;
Stage1 protokol/çekirdek dosyaları değişmedi. Model/seed/λ/eşik seçimi yok.

21 checkpoint × 10 sabit validation tanığı: q birebir eş, bağımsız NumPy FK
ve tarihsel Pinocchio FK G0 toleranslarında. C1-05 batch tanığına ek olarak
tek sorgu yürütücüsü özgün c105.infer ile aynı girdide birebir doğrulandı.
C1-04 üç conditioned checkpoint mevcut on tanığın tek sorgu çıktısını tekrarlar.
Tam sonuç ve kaynak kod hashleri preflight.json içinde.

59 sentetik/negatif test PASS, fail/error/skip0. Testler final sorgu içermez.
Yeni wrapper; onay, sıra/kimlik, exact pose/q_current, hash, deadline, eksik
payda, tekrar, null metrik, fractional baseline bootstrap, bozuk girdi ve
overwrite engellerini kontrol eder. runtime-tests.xml ve komut logları kanıttır.

İlk C1-04 eski genel wrapper çağrısı yalnız tarihsel .gitattributes SHA'sını
güncel depo politikasıyla karşılaştırdığı için exit1 verdi. Bu dosya sonraki
C1-05/C1-06 kanıt yollarını içerdiğinden doğal olarak değişmiştir. Tarihsel
dosya/kabul düzenlenmedi. Yeni preflight substantive checkpoint/config/robot,
normalizasyon, kaynak ve tanık q/FK'yı doğrudan doğrular. Teknik kapsam
daraltılmadı; eski başarısız komut logu korundu. Nihai açılış bu kontrolden sonra.

C1-01 raw, dondurulmuş sorguya exact identity ile bağlanacak. Aynı NumPy
denetleyici bütün saklanmış baseline q adaylarına uygulanacak; Profile A/B ve
limit kararları tarihsel sonuçla aynen eşleşmeli. Tarihsel Linux ile Windows
residual farkı ayrıca maksimum metre/derece olarak kaydedilecek; tarihsel
raw değiştirilmez. Bu yeni çapraz platform audit, eski Linux sayısal
değerlerinin byte düzeyinde yeniden üretimi diye sunulmaz (ADR-008 kapsamı).

Kampanya: final-001. Raw data/generated/C1-06/final-001 altında LOCAL_ONLY.
21 × 12.000 × 5 = 1.260.000 neural ölçüm; historical baseline 600.000 satır.
Yükleme/ısınma süreleri ayrı, tek sorgu CPU uçtan uca ölçüm. GPU NOT_RUN.
Uzak arşiv NOT_CONFIRMED. Disk açılış gözleminde yaklaşık 308,8 GB boş;
tepe RAM/süre gerçek koşu sonrasında kaydedilir, önceden başarı varsayılmaz.
