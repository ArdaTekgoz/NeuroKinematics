# C1-06 bağımsız nihai sonuçlar

Ölçüm: 8 Ekim 2026 · Teslim: 9 Ekim 2026 · Ön kayıt r1 · T-C05 PASS / araştırma değerlendirmesi tamam.

**H2: REJECTED**. Değerler aynı 12.000 query_id üzerinde, üç sabit eğitim seed'inin eşit ortalamasıdır. Güven aralığı bu üç seed koşulunda query-root örnekleme belirsizliğidir; seed popülasyonuna genelleme aralığı değildir.

## Birincil H2: E-C05/FK_TANH − E-C03/Q

| Küme | Tekil N | Fark ve %95 CI (yüzde puanı) | Ön kayıt hedefi |
|---|---:|---|---|
| main | 10000 | +0.000 [+0.000, +0.000] | ≥−1 yp |
| boundary | 1000 | +0.000 [+0.000, +0.000] | ayrı rapor |
| singularity | 1000 | +0.000 [+0.000, +0.000] | ayrı rapor |
| hard_equal_weight | 2000 | +0.000 [+0.000, +0.000] | ≥+2 yp; 50/50 ağırlık |

| Eğitim seed | Main fark/CI (yp) | Zor eşit ağırlık fark/CI (yp) |
|---|---|---|
| 2026100201 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| 2026100202 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| 2026100203 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

Her grupta aynı query/root tek kez paydaya girer. Beş süre geçişi bağımsız sorgu değildir. 10.000 PCG64 bootstrap tekrarı, seed 2026100806, percentile %95 CI kullanıldı. Üç seed ayrı raporlandı; daha iyi seed seçilmedi.

Sıfır empirik fark için [0,0] aralığı gerçek popülasyonda başarı olasılığının tam sıfır olduğunu kanıtlamaz. Karar, önceden belirlenen +2 yüzde puanlık zor-küme artışı hedefinin bu final örneklemde desteklenip desteklenmediğine ilişkindir.

## Bütün neural modeller

| Deney / model | Seed | N | Profil A / B başarı | Limit dışı | Poz medyan m / P95 m | Yönelim medyan ° / P95 ° |
|---|---:|---:|---|---:|---|---|
| E-C03/Q | 2026100201 | 12000 | 0 / 0 | 925 | 0.213 / 0.983 | 73.935 / 169.161 |
| E-C03/FK | 2026100201 | 12000 | 0 / 0 | 1435 | 0.216 / 0.622 | 27.223 / 80.304 |
| E-C03/Q | 2026100202 | 12000 | 0 / 0 | 600 | 0.221 / 0.993 | 77.979 / 169.790 |
| E-C03/FK | 2026100202 | 12000 | 0 / 0 | 717 | 0.167 / 0.465 | 26.568 / 86.080 |
| E-C03/Q | 2026100203 | 12000 | 0 / 0 | 645 | 0.221 / 0.995 | 78.252 / 169.642 |
| E-C03/FK | 2026100203 | 12000 | 0 / 0 | 960 | 0.174 / 0.505 | 27.842 / 97.458 |
| E-C04/FK | 2026100201 | 12000 | 0 / 0 | 1946 | 0.367 / 0.779 | 50.655 / 125.890 |
| E-C04/FK_LIMIT | 2026100201 | 12000 | 0 / 0 | 1735 | 0.367 / 0.792 | 51.755 / 127.806 |
| E-C04/FK | 2026100202 | 12000 | 0 / 0 | 717 | 0.167 / 0.465 | 26.568 / 86.080 |
| E-C04/FK_LIMIT | 2026100202 | 12000 | 0 / 0 | 982 | 0.178 / 0.516 | 28.051 / 97.494 |
| E-C04/FK | 2026100203 | 12000 | 0 / 0 | 1799 | 0.350 / 0.757 | 70.189 / 161.203 |
| E-C04/FK_LIMIT | 2026100203 | 12000 | 0 / 0 | 1624 | 0.350 / 0.763 | 70.785 / 162.500 |
| E-C05/FK | 2026100201 | 12000 | 0 / 0 | 1946 | 0.367 / 0.779 | 50.655 / 125.890 |
| E-C05/FK_TANH | 2026100201 | 12000 | 0 / 0 | 0 | 0.288 / 0.741 | 62.637 / 152.483 |
| E-C05/FK | 2026100202 | 12000 | 0 / 0 | 717 | 0.167 / 0.465 | 26.568 / 86.080 |
| E-C05/FK_TANH | 2026100202 | 12000 | 0 / 0 | 0 | 0.142 / 0.422 | 26.016 / 92.776 |
| E-C05/FK | 2026100203 | 12000 | 0 / 0 | 1799 | 0.350 / 0.757 | 70.189 / 161.203 |
| E-C05/FK_TANH | 2026100203 | 12000 | 0 / 0 | 0 | 0.276 / 0.734 | 60.627 / 152.398 |
| E-C01/conditioned | 2026100201 | 12000 | 0 / 0 | 925 | 0.213 / 0.983 | 73.935 / 169.161 |
| E-C01/conditioned | 2026100202 | 12000 | 0 / 0 | 600 | 0.221 / 0.993 | 77.979 / 169.790 |
| E-C01/conditioned | 2026100203 | 12000 | 0 / 0 | 645 | 0.221 / 0.995 | 78.252 / 169.642 |

Poz/yönelim dağılımları yalnız finite + limit içi q üzerinde hesaplanır. Invalid q sessiz clamp edilmez; geometri eksikliği ve coverage her model/subgroup için results.json içinde. Başarı oranı her zaman bütün sorgu paydasındadır. Tam P99, başarılı-altküme ve timeout dağılımları aynı dosyadadır.

## İkincil ablasyonlar

| Keşifsel karşılaştırma | Main fark/CI (yp) | Zor fark/CI (yp) |
|---|---|---|
| E-C03/FK − E-C03/Q | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| E-C04/FK_LIMIT − E-C04/FK | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| E-C05/FK_TANH − E-C05/FK | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| E-C03/Q − E-C01/conditioned | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

İkincil CI nominaldir; çoklu karşılaştırma düzeltmesi yapılmadı. E-C01 conditioned ve E-C03 Q aynı validation çıktısına sahip tekrarlı kontroldür. E-C03 200 epoch; E-C04 26/200/26; E-C05 27/200/28. FK_TANH best epochları 7/186/8. Birincil FK_TANH−Q farkı FK kaybı ve bounded head birleşimidir; yalnız FK veya yalnız limit mekanizmasına atfedilemez.

FK kaybı E-C03’te yönelim medyanını Q’nun 73,9–78,3° aralığından
26,6–27,8° aralığına indirdi; konum etkisi üç seedin ikisinde iyileşti, birinde
kötüleşti. E-C04 limit ihlalini iki seedde azalttı (1946→1735, 1799→1624),
birinde artırdı (717→982). FK_TANH üç seedde de limit ihlalini sıfırladı;
fakat poz/yönelim toleranslarını birlikte sağlayan sorgu üretmedi. Hata
medyanındaki iyileşme, geçerli IK çözümü artışı olarak yorumlanamaz. Seed2’nin
daha düşük hatası nedeniyle diğer seedler çıkarılmadı.

## FK_TANH hata dağılımları

| Seed | Alt küme | N | Profil A | Poz medyan / P95 (m) | Yönelim medyan / P95 (°) |
|---|---|---:|---:|---|---|
| 2026100201 | main/local | 5000 | 0 | 0.260 / 0.700 | 62.437 / 150.853 |
| 2026100201 | main/wide | 5000 | 0 | 0.307 / 0.754 | 63.744 / 152.617 |
| 2026100201 | boundary/local | 500 | 0 | 0.374 / 0.831 | 71.719 / 161.362 |
| 2026100201 | boundary/wide | 500 | 0 | 0.371 / 0.962 | 77.354 / 163.346 |
| 2026100201 | singularity/local | 500 | 0 | 0.221 / 0.584 | 44.897 / 142.537 |
| 2026100201 | singularity/wide | 500 | 0 | 0.307 / 0.746 | 53.831 / 145.870 |
| 2026100202 | main/local | 5000 | 0 | 0.121 / 0.407 | 22.909 / 84.139 |
| 2026100202 | main/wide | 5000 | 0 | 0.160 / 0.427 | 29.636 / 97.676 |
| 2026100202 | boundary/local | 500 | 0 | 0.179 / 0.524 | 28.466 / 116.162 |
| 2026100202 | boundary/wide | 500 | 0 | 0.179 / 0.473 | 36.984 / 110.957 |
| 2026100202 | singularity/local | 500 | 0 | 0.086 / 0.347 | 16.702 / 64.247 |
| 2026100202 | singularity/wide | 500 | 0 | 0.125 / 0.386 | 24.776 / 95.894 |
| 2026100203 | main/local | 5000 | 0 | 0.256 / 0.690 | 60.416 / 151.757 |
| 2026100203 | main/wide | 5000 | 0 | 0.292 / 0.744 | 61.639 / 150.297 |
| 2026100203 | boundary/local | 500 | 0 | 0.345 / 0.806 | 67.785 / 162.636 |
| 2026100203 | boundary/wide | 500 | 0 | 0.361 / 0.910 | 73.847 / 165.088 |
| 2026100203 | singularity/local | 500 | 0 | 0.209 / 0.585 | 41.909 / 142.941 |
| 2026100203 | singularity/wide | 500 | 0 | 0.269 / 0.714 | 52.029 / 143.642 |

Her modelin en kötü 20 konum ve 20 yönelim sorgusu, query/root kimlikleri ve raw q değerleriyle results.json içinde saklanır. Limit yakını, küçük sigma_min ve |position z-score|>3 tanıları keşifsel alt kümelerdir; üyelik çakışmaları birincil tekil paydayı değiştirmez. C1-02 teacher başarısızlıkları final benchmark satırı değildir; bu benchmark öğretmen etiketi kullanmadığından label/teacher alanları NOT_APPLICABLE. Eğitimdeki teacher eksikliği ve benchmark ±0,05 rad / eğitim ±0,1 rad local farkı genelleme sınırlarıdır.

## Aynı sorgularda beş tarihsel baseline

| Yöntem | Bütçe ms | Tekil N | A geometri % | A deadline % | B deadline % | Tarihsel P50 / P95 ms |
|---|---:|---:|---:|---:|---:|---|
| dls/default | 10 | 12000 | 59.457 | 55.997 | 55.997 | 5.529 / 10.933 |
| dls/default | 50 | 12000 | 69.565 | 69.187 | 69.022 | 5.504 / 50.879 |
| kdl/default | 10 | 12000 | 93.672 | 93.253 | 93.253 | 1.103 / 10.947 |
| kdl/default | 50 | 12000 | 98.773 | 98.760 | 98.760 | 1.097 / 12.991 |
| trac_ik/speed | 10 | 12000 | 99.997 | 99.997 | 99.997 | 1.162 / 1.849 |
| trac_ik/speed | 50 | 12000 | 99.998 | 99.998 | 99.998 | 1.162 / 1.848 |
| pick_ik/local | 10 | 12000 | 62.625 | 62.625 | 62.625 | 1.262 / 10.841 |
| pick_ik/local | 50 | 12000 | 62.625 | 62.625 | 62.625 | 1.299 / 51.036 |
| pick_ik/global | 10 | 12000 | 1.283 | 0.430 | 0.430 | 36.749 / 40.951 |
| pick_ik/global | 50 | 12000 | 98.212 | 89.007 | 89.007 | 42.566 / 57.572 |

Her satır 12.000 sorgu × beş historical ölçüm geçişidir. Geometri başarıları bütün adayların aynı bağımsız NumPy FK denetiminden gelir. Candidate olmayan kayıtlar no_candidate=true ile sıfır geometri başarısıdır; türetilmiş raw failure_class=INVALID_SHAPE bu durumda doğrulayıcıya boş aday verilmesini ifade eder, solverın bozuk boyutlu çıktı ürettiği iddiası değildir. Asıl common_status ve no_candidate ayrı korunmuştur.

| FK_TANH − tarihsel baseline | Bütçe ms | Main fark/CI (yp) | Zor fark/CI (yp) |
|---|---:|---|---|
| dls/default | 10 | -59.796 [-60.750, -58.854] | -57.760 [-59.890, -55.650] |
| dls/default | 50 | -69.770 [-70.684, -68.864] | -68.540 [-70.550, -66.490] |
| kdl/default | 10 | -94.754 [-95.062, -94.438] | -88.260 [-89.390, -87.060] |
| kdl/default | 50 | -99.376 [-99.506, -99.238] | -95.760 [-96.540, -94.930] |
| trac_ik/speed | 10 | -99.998 [-100.000, -99.994] | -99.990 [-100.000, -99.970] |
| trac_ik/speed | 50 | -100.000 [-100.000, -100.000] | -99.990 [-100.000, -99.970] |
| pick_ik/local | 10 | -63.270 [-64.200, -62.330] | -59.400 [-61.550, -57.250] |
| pick_ik/local | 50 | -63.270 [-64.200, -62.330] | -59.400 [-61.550, -57.250] |
| pick_ik/global | 10 | -1.374 [-1.476, -1.272] | -0.830 [-1.000, -0.660] |
| pick_ik/global | 50 | -98.678 [-98.840, -98.514] | -95.880 [-96.480, -95.260] |

Bu CI'lar query başına beş tekrarın ortalama başarısı ile neural 0/1 başarısı arasında, aynı query-root üzerinde eşlidir. Keşifsel ve üç sabit seed koşulludur; eşlenmeyen sorgu yoktur. Tarihsel timeout/çözülememe erişilemezlik kanıtı değildir.

## CPU süreleri ve yorum sınırı

| FK_TANH seed | Yükleme s | Warmup s | Tam-payda P50 / P95 / P99 ms | Başarılı ölçüm N |
|---|---:|---:|---|---:|
| 2026100201 | 0.013650 | 0.001911 | 0.360 / 0.617 / 0.828 | 0 |
| 2026100202 | 0.014342 | 0.001401 | 0.364 / 0.633 / 0.871 | 0 |
| 2026100203 | 0.013130 | 0.002062 | 0.361 / 0.611 / 0.833 | 0 |

Windows CPU tek sorgu: giriş hazırlığı + ileri geçiş + q dönüşümü + bağımsız FK doğrulama. Model yükleme ve 20 validation warmup ayrı; kayıt I/O'su ölçüm dışında. Beş geçişte q/geometri birebir tekrarladı. Ubuntu/ROS baseline farklı host/runtime/IPC/kaynak politikasına aittir; tablolar hız üstünlüğü kanıtı değildir. Başarılı-altküme N=0 ise onun süresi N/A'dır. GPU throughput NOT_RUN.

## Kanıt ve devir

1.260.000 neural ölçüm + 600.000 baseline doğrulama satırı; 21 model, üç seed, 12.000 tekil sorgu. Yeni yerel raw paketi 2,331,485,455 bayt. [Raw manifest](final-001/raw-manifest.json), [sonuçlar](final-001/results.json), [kabul](final-001/acceptance.json), [C1-07 devri](final-001/C1-07-handoff.json). Checkpoint/ham veri LOCAL_ONLY, uzak arşiv NOT_CONFIRMED. Collision NOT_CHECKED; fiziksel robot, G1 ve v1.0.0 etiketi bu görevde yok.
