# Negatif ve mutasyon matrisi — r1

Hepsi **NOT_RUN**; **24 planlı hata sınıfı**, parametrelemeyle gerçek test sayısı
artabilir ve Stage 2 raporunda açık verilir. Mutasyon: üretim Torch kernel/adapter
veya doğrulayıcısının geçici kopyasına küçük diff, aynı normal test suite'i;
orijinal dosya/robot korunur. AST/source veya gerçek dependency-boundary
monkeypatch uygulanır; mutantın kendisini onaylayan mock oracle yasaktır.
Önce sağlam kod geçer, sonra ilgili mutant beklenen assertion ile düşer.
Syntax/import hatası 'killed' sayılmaz; survived/error ayrı kaydedilir.

| ID | Hata / enjeksiyon yeri | Öldüren kontrol / beklenen sonuç |
|---|---|---|
| M01 | Kernel q indeksinde 2 sütunu değiştir | Mixed ve tüm FK / column gradient eşliği FAIL |
| M02 | q'yu derece olarak çarp/çevir | Mixed T-C01 FAIL; dış API units=deg açık ValueError |
| M03 | Joint_1 axis işaretini tersle | T-C01 ve açısal Jacobian FAIL |
| M04 | origin @ motion yerine motion @ origin | A2/A4 analitik poz ve T-C01 FAIL |
| M05 | Fixed TCP'yi atla veya flange döndür | A1/A4/A5 R/p normu FAIL |
| M06 | World FK'yi relatif base sonucu diye döndür | A1/A4 mount invariance ve bağımsız Pinocchio relatif eşliği FAIL |
| M07 | Base inverse'i iki kez uygula/yanlış frame seç | A1/A4 manuel relatif T FAIL; eksik frame açık hata |
| M08 | Üretim sonucu zorla float32 yap, f64 diye geri çevir | Dtype sözleşmesi / f64 T-C01 FAIL |
| M09 | Batch satırlarını döndür veya ilkini broadcast et | Id koruma/singleton/reverse ve block Jacobian FAIL |
| M10 | q→NumPy→Torch ile hesapla | p/R backward graph bağlantısı veya import sınırı FAIL |
| M11 | q.detach veya no_grad ile kernel hesapla | requires_grad/grad_fn ve backward FAIL |
| M12 | Graph'a zarar veren in-place tensor overwrite | Backward anomali/versiyon veya FD FAIL; runtime traceback saklanır |
| M13 | Straight-through doğru T ama yanlış permütasyonlu backward | T-C01 geçse bile bileşen bazlı T-C02 / J sütunları FAIL |
| M14 | Zero/constant gradient hook/custom backward | 12 bileşen FD, Jacobian, nonconstant q ve sensitivity FAIL |
| M15 | Forward'a veya backward'a NaN/Inf enjekte et | Her çıktı/gradyan finite kapısı FAIL; tüm satırlar saklanır |
| M16 | Limit yakınında merkez stencil'i zorla veya q±h clip et | Exact/near için stencil log ve bağımsız 2. derece edge FD FAIL |
| M17 | Oracle olarak aynı Torch kernel'i kullan | Oracle import/provenance kontrolü + bozuk axis mutantı bağımsız oracle karşısında FAIL; paylaşılmış backend suite kabul edilmez |
| M18 | Robot/TCP/config/sample dosyası bir bayt değişsin | Preflight SHA uyuşmazlığı; ileri hesap başlamadan FAIL |
| M19 | Quaternion xyzw/wxyz veya işaret eşdeğerliğini boz | Mevcut metrik gerçek kodunun geçici mutantı E03 / F0-03 metrics FAIL; temel matrix FK'de quaternion yok |
| M20 | Strict '>'/'<' veya yuvarlanmış geniş float32 limit | Exact float64 kabul, tek ULP dışarı reddi; float32 gerçek sınır içi kontrolü FAIL |
| N21 | Tensor shape/rank/empty/dtype/type/nonfinite/limit dışı | Her biri açık TypeError/ValueError; kısmi batch geçişi yok |
| N22 | Robot id, joint order, units metadata yanlış/eksik | Dış API çağrısı TypeError/ValueError; sayılardan otomatik birim tahmini yok |
| N23 | Eksik/bozuk limit/axis, unknown/prismatic/continuous/mimic, cycle/disconnected/ambiguous frame | Constructor açık ValueError; fixture üzerinde parser/validation testi |
| N24 | Eksik raw satır, duplicate id, yanlış dtype etiketi, eksik failed kayıt, bozuk raw hash | Evidence kabul doğrulayıcısı FAIL; başarı özeti tek başına yeterli değil |

M06 özel ayrım: Torch'un doğrudan base→TCP zincirini hesaplaması upstream fixed
world→base mount'ını matematiksel olarak iptal eder; bu doğru davranıştır.
Mutant dünya sonucunu base sonucu diye etiketlemelidir. Upstream mount'ın etkisiz
olmasını yanlışlık sayan bir test yazılmaz. TCP offset/rotation ve base'in
world'de nonidentity olması aynı A1/A4 fixture'da bağımsız denetlenir.

Mutation sonuç alanları: id, baseline test PASS, mutant source SHA/diff,
enjeksiyon dosya/satır, killing test id, expected failure, observed failure,
exit, stdout/stderr, killed/survived/error. 24 sınıfın hiçbiri kaybolamaz;
uygulanmayan vaka açık karar ister ve tam kabul sağlanmış sayılmaz.
