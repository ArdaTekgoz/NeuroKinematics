# Deney veya uygulama kaydı

Kimlik: RUN-20261011-C107-CLOSURE
Durum: COMPLETE / T-C06 PASS / G1 RESEARCH ACCEPTED
Görev ve gereksinim: C1-07 / REQ-C06 / T-C06; akademik negatif sonuç raporu
Tarih ve sorumlu: 11 Ekim 2026 / Arda Tekgöz; uygulama ve analiz desteği Codex

## Soru ve değişiklik

Altı sabit araştırma adayı temiz kaynak kopyasında, yeniden kurulmuş kilitli
ortamda aynı çıkarım ve bağımsız değerlendirme sonucunu üretiyor mu?
Yeni portable inference modülü, 48 sorguluk stratified witness, bozuk girdi
ve hash kontrolleri, taze ortam kurucusu eklendi. PROTOCOL.md ön kayıttır.
Akademik negatif sonuç metni, PDF ve D9'un altı hücresinden iki grafik
üretildi. Başarı eşikleri ve tarihsel kaynaklar değiştirilmedi.

## Tekrar üretim

Başlangıç commit'i 46d80f3. Kullanıcının STATUS ve TRACEABILITY değişiklikleri
ayrı korunur. Kaynak dondurma commit'i temiz çalışma start.json kaydında
verilecektir. pixi.lock ve requirements-win-cu128.lock aynı kalır.
Robot, normalizasyon, config ve checkpoint SHA'ları preparation manifestinde;
eğitim seed'leri FK_TANH 2026100201/02/03, LOCAL_RAW 2026100901/02/03.
Windows 11 / Ryzen 7 250 / RTX 5060 Laptop 8 GB / 24 GB RAM.
Witness CPU, thread 1, batch 48; decoder regresyonu CPU ve CUDA.

## Test ve ham kanıt

- commands/003-witness-record: 48 validation sorgusu × 6 aday referansı, PASS.
- commands/004-local-tests: mevcut ortamda 11 test PASS.
- commands/005-portable-tests: checkpoint hash mutantı yerel ağırlık gerektirmeyecek
  şekilde taşınabilir yapıldı; 11 test PASS.
- Temiz tekrar: henüz NOT_RUN; aşağıya gerçek sonuç eklenecek.
- PDF beş sayfaya render edildi; her sayfa görsel olarak kontrol edildi:
  kesilme, taşma veya eksik Türkçe glif gözlenmedi. Figür değerleri results.json
  üzerinden figure-data.csv'ye çıkarılır. Kaynak SHA'ları evidence-index.json.
- Artifact üretimi bundled Python/ReportLab ve matplotlib 3.10.8 ile yapıldı;
  bu araçlar Core runtime'ına eklenmedi. İlk üretimde matplotlib eksikliği
  giderildi; bilimsel deney sonuçlarını etkilemez.

## Sonuç ve yorum

Mevcut ortam kontrolleri PASS, T-C06 henüz PENDING. Akademik rapor yeni
bağımsız test veya H1 sonucu değildir. Eski final raw analiz için açılmadı;
C1-06R yeni final NOT_CREATED. Yeni eğitim NOT_RUN.

## Sonraki adım

Kaynağı commit ile sabitle; temiz clone'da reproduce_c107.py çalıştır.
Gerçek sonuçlara göre nihai model kartı, G1 kararı, devir ve ara verme
paketini tamamla. Hybrid uygulaması başlatılmayacak.

## Nihai uygulama ve kabul · 11 Ekim 2026

Yukarıdaki NOT_RUN/PENDING ve sonraki adım ifadeleri kaynak dondurma anının
kaydıdır. Aşağıdaki sonuçlar bu planı tamamlar:

- 006-clean-reproduction: temiz c7798ef clone, yeni Pixi/venv, SHA-pinned
  overlay kurulum, pip check, runtime kimliği, witness ve regresyonlar PASS.
  UTC 23:43:03.949596–23:46:42.878442; toplam komut 218,93 saniye.
  İç protokol start–complete yaklaşık 218,55 saniye. İstanbul tarihi 11 Ekim.
- 48 sorgu × 6 model = 288 q/FK çıktısı birebir; 11 handoff, 79 FK,
  31 physics ve 6 CPU/CUDA decoder testi = 127 PASS; 0 fail/error/skip.
  NaN mutantının beklenen NumPy determinant uyarısı korunur.
- 007-local-archive: 607 dosya, 6087040251 ham bayt, 1046673229 ZIP baytı;
  her üye SHA doğrulandı. 008-restore-check: boş klasöre tamamı geri
  yüklendi ve yeniden SHA doğrulandı. Arşiv opak bayt korumasıdır;
  eski sealed/final dosyalar tekrar analiz edilmedi.
- 009-closure-integrity: 122 frozen kaynak, D3–D9 ve hazırlık kayıtları,
  455 eski C1-06R teslimi ve 16 hazırlık teslimi değişmez; temiz 11 komutun
  stdout/stderr SHA ve exit kodları PASS.
- Nihai MODEL_CARD, G1_DECISION, acceptance ve HYBRID_HANDOFF hazır.
  Akademik PDF, kaynak Markdown, iki PNG/SVG grafik ve kaynak CSV/hash
  indeksi hazır. LinkedIn metni taslaktır; yayımlanmadı.
- STATUS/TRACE içindeki önceki kullanıcı değişiklikleri ayrı kopyalanır
  ve yalnız bu görevin ekleri commit'e alınır. Kaynak ve belge sürümleri
  ayrıdır; release/tag oluşturulmaz. Git push sonucu ayrı teslim makbuzunda.

REQ-C06 → portable inference + nihai model kartı → T-C06 127 test ve
288 witness → clean/complete.json, witness-result.json, integrity.json.
Core/G1 araştırma kapanışı PASS / ACCEPTED. Ürün NOT_MET, H2 REJECTED,
direct IK NO_GO. Yeni eğitim ve bağımsız final NOT_RUN/NOT_CREATED.

## Devam noktası ve kalan işler

H2-01 sıradadır; Hybrid NOT_STARTED, H1 NOT_MEASURED. Geri dönüş tarifi
docs/records/CORE_RESUME.md; akademik sınırlar negatif sonuç raporunda.
Uzak büyük dosya arşivi ve bağımsız aygıt yedeği NOT_CONFIRMED;
yerel ZIP'in ikinci bir ortama kopyalanması kullanıcıya kalan saklama işidir.
Makaleye kapsamlı literatür yerleştirme, kalıcı artifact erişimi ve insan
bilimsel değerlendirmesi yapılmadan hakemli yayın iddiası kurulmaz.
