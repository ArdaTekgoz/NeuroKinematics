# C1-05 model kararı ve C1-06 devri

8 Ekim 2026 · T-C04 PASS / deneysel karşılaştırmalar COMPLETE / doğrudan IK NO-GO

C1-04'ün her seed'de0/3600 olan Profil A sonucu değişmedi. FK terimleri yönelim
hatasını azaltırken limit ihlalini artırdı; limit cezasının etkisi karışıktı.
Tanh ham sınır ihlalini sıfırladı fakat bütün üç seed'de Profil A/B yine0/3600.
Bu sonuç kayıp azalmasının operasyonel IK başarısı olmadığını gösterir.

Ön kayıtlı sıralama (ortalama Profil A, geçersiz sayısı, tam-payda konum/yönelim
medyanı, q-loss, basitlik) **FK_TANH** ailesini C1-06 araştırma adayı seçti.
Üç seed birlikte devredilir; tanık seed baştan belirlenen2026100201'dir,
daha iyi görünen seed2 sonradan tek model olarak seçilmez. E-C05 best epochları
7/186/8; eğitim epochları27/200/28. Seçim hâlâ en düşük validation Lq'dur.
Farklı erken durdurma bütçeleri nedeniyle dört aileyi tek bir eşit-gerçekleşen-
compute nedensel kıyas diye sunma; etki yorumu kendi eşli deneylerine aittir.

[Hashli devir manifesti](C1-06-handoff.json) model/config/veri/normalizasyon,
checkpoint yol/byte/SHA, erişim ve seçme protokolünü içerir. Bütün18 best ve18 last
checkpointin envanteri [sonuç auditinde](results-audit.json). Ağırlıklar LOCAL_ONLY;
uzak arşiv NOT_CONFIRMED. Temiz checkout/yeni kilitli ortamda18 checkpointten
180 çıkarım ve geçerli FK pozları birebir tekrarlandı. Taze ortamda eğitim NOT_RUN.

C1-06 test ve10000 benchmark SEALED_NOT_RUN. H2'nin bağımsız nihai kararı C1-06'ya
aittir; burada validation üzerinde olumlu H2 veya üstünlük ilan edilmez. Kaynak
hataları bulunursa eğitim/kanıt kapısı tekrar kapanır; eşikler düşürülmez.
E-C06/07/08, Res-MLP/curriculum SKIP olarak korunur. Yeni mimari/λ araması bu
kapanışın parçası değildir. G1 ve v1.0.0 sürüm etiketi verilmedi.
