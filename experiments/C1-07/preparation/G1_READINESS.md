# G1 hazırlık matrisi ve yürütülebilir sonraki işler

10 Ekim 2026 · Durum PREPARATION_COMPLETE / G1_OPEN / T-C06_PENDING.

## Mevcut kanıtlar

| Gereksinim | Kanıt | Durum ve anlamı |
|---|---|---|
| G0/robot/TCP | load_robot frozen4 + hazırlık registration/manifest | Hash PASS; Foundations değişmedi |
| C1-01/T-C00 baselinelar | ../../C1-01/RUN-20260928-T-C00-acceptance.md | Tarihsel kabul; bu tur solver benchmarkı yeniden çalışmadı |
| C1-02/T-C07 veri | ../../C1-02/acceptance.json | Tarihsel PASS; tanılarda train/validation ayrımı korunmuş |
| C1-03/T-C01–02 FK/gradyan | ../../C1-03/stage2/acceptance.json | Tarihsel PASS; bu tur inference bağımsız FK de PASS |
| C1-04/T-C03 | ../../C1-04/stage2/acceptance.json | PASS; doğrudan IK NO_GO |
| C1-05/T-C04 | ../../C1-05/stage2/acceptance.json | PASS; ürün başarısı yok |
| C1-06/T-C05/H2 | ../../C1-06/stage2/final-001/acceptance.json | PASS / H2 REJECTED / direct NO_GO |
| C1-06R | C1-06R-closure.json, ADR-025 | Araştırma sonlandırıldı; ürün NOT_MET; yeni final yapılmadı |
| C1-07 aday kimliği | handoff-manifest.json, registration.json | Altı checkpoint mevcut ve SHA doğrulanmış |
| C1-07 model kartı | MODEL_CARD_DRAFT.md | Hazırlık taslağı; runtime doğrulamasından sonra nihai hale gelecek |
| C1-07/T-C06 | preparation-audit.json | Mevcut ortam21600 denetim PASS; taze ortam testi NOT_RUN |
| G1 | Henüz nihai karar yok | OPEN; Hybrid geçişi yetkisi üretmez |

Tablodaki eski kabul kayıtları tarihsel kanıttır; bu tur bütün eski testleri
çalıştırmış gibi gösterilmez. C1-05'in clean/complete.json kaydı geçmişte
temiz ortam tekrarı olduğunu gösterir; yeni altı-aday paketinin T-C06 yerine
kullanılmaz. Devir kapsamı ve kaynakların tamlığı yeni paketle doğrulanmalıdır.

## T-C06/G1 için kalan iş paketi

1. **Paket kimliği:** Bu hazırlığın commit'i sabitlenir; aday manifesti,
   inference kaynağı, feature/decoder sözleşmesi ve robot/scaler/checkpoint
   hashleri taşınabilir pakete alınır. Ağırlıklar normal Git'e konmaz.
2. **Bağımsız tekrar girdisi:** Train/validation'dan sabit, family/mode
   kapsamı belgeli çıkarım örnekleri hazırlanır. q_current ve target pose
   taşınır; model girdisine teacher/root/split alanı girmez. Beklenen q,
   bağımsız FK ve A/B sonuçları sabitlenir. Eski final raw açılmaz.
3. **Temiz tekrar:** Yeni boş checkout ve taze ortamda kilitlerden kurulum;
   eski .venv/.pixi kopyalanmaz. Altı checkpoint weights_only yüklenir;
   örnek çıkarım ve değerlendirme yeniden yapılır. CPU/thread1/batch sınırı
   sabitlenir. Exact eşlik uygun değilse fark toleransları çalışmadan önce
   sayısal sözleşmeden türetilir; sonuç gördükten sonra genişletilmez.
4. **Regresyon ve başarısızlık kontrolleri:** Seçili inference/decoder/FK
   testleri; yanlış robot/scaler/ağırlık SHA ve bozuk girişlerin reddi;
   limit dışı tahminin başarılı sayılmaması. Legacy test klasörlerinin
   conftest import çakışması nedeniyle gerektiğinde ayrı süreçler kullanılır.
5. **Nihai model kartı:** Bu taslağa gerçek temiz ortam komut/log/sonuçları,
   desteklenen runtime ve kalan kısıtlar eklenir. LOCAL_RAW local-only
   kapsamı korunur; fail satırlar silinmez. Remote archive durumu ayrıca
   doğrulanmadan CONFIRMED yazılmaz.
6. **G1 kararı:** Kritik doğruluk/tekrar üretim ve gerekli kapsam tamlığı
   geçtiyse araştırma kapanışı gerekçeli olarak değerlendirilir. Direct IK
   NO_GO ve H2 REJECTED görünür kalır. Başarısız temiz tekrar varsa G1 açık
   kalır; yazılım kusuru bulunan aday Hybrid'e kabul edilmez.

Bu altı iş bu hazırlıkta çalıştırılmış sayılmaz. Eski
scripts/reproduce_c105.py ve c105_witness.py kullanılabilir tasarım
referansıdır; yalnız eski18 model/witness kapsamını yürütürler. Yeni altı
aday için hazır komutmuş gibi kopyalanmamalı. Yeni C1-07 tekrar CLI'si ve
taşınabilir fixture gerekli; oluşturulmadan çalıştırılmış komut yazılmadı.

## H1 için G1 sonrası dondurulacaklar

- Birincil FK_TANH üç seed ve secondary LOCAL_RAW üç seed rolleri; model
  taramasından en iyi sonucu seçip tek ana sonuç gibi sunmama.
- Aynı sayısal motor, residual/Jacobian, tolerans ve adım kabulü; toplam
  10/50ms bütçe, batch1/aynıCPU/thread, en az10000 sorgu ve5 zaman geçişi.
- q_current/merkez/neural/klasik restart kontrolleri; neural tüm maliyeti
  dahil. Projeksiyon/restart/deadline politikaları ön kayıtlı.
- Validation araştırma verisidir. Yeni H1 değerlendirme kümesinin kimliği,
  train/validation/root ayrımı ve erişim protokolü model seçimi bitmeden
  yazılır. Eski finali yeni aday seçmek için kullanma.
- H1 eşli bootstrap ve seed değişkenliği; mutlak başarı/geçersiz/deadline
  ve local/wide/zor gruplar. Hiçbir ölçüm bu hazırlıkta yapılmadı.

## İş bölümü ve aşama sırası

Kod, paket, tekrar üretim, denetim ve analiz Codex tarafından yürütülür.
Uzun çalışma gerekirse ölçülmüş bütçeli tek komut kullanıcıya teslim edilir;
şu anda yeni eğitim talebi yok. Kullanıcının mevcut üç aşamalı talebinde
ikinci aşama burada tamamlanır; üçüncü aşama makale için başarısızlık
raporudur ve ayrı komut bekler. T-C06/G1 kapama ve Hybrid uygulaması,
rapor aşamasının içine sessizce eklenmez.
