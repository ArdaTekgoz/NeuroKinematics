# Deney veya uygulama kaydı

Kimlik: RUN-20261011-C107-CLOSURE
Durum: IN_PROGRESS / SOURCE_FREEZE
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
