# Deney veya uygulama kaydı

Kimlik: RUN-20261011-CORE-PROGRESS-COMM  
Durum: COMPLETE / DRAFT_READY / NOT_PUBLISHED  
Görev ve gereksinim: İlk iki fazın kanıta dayalı LinkedIn metni/görselleri;
REQ-C06 iletişim eki, H2 plan etkisi açıklaması  
Tarih ve sorumlu: 11 Ekim 2026; kullanıcı isteğiyle Codex

## Soru ve değişiklik

Foundations ve Core kapanışları, neural araştırmadaki engel ve yeni
geometri başarısı doğru biçimde nasıl anlatılır? Kullanıcının "yeni neural
model" varsayımı düzeltildi: C1-06S learned=false, eski ağlar yeniden
eğitilmedi. G0 kabulü, G1 araştırma kabulü ve geometri ürün başarısı ayrı
gösterildi. Başarı eşikleri/kod/roadmap değiştirilmedi.

Yeni dosyalar: docs/publication/core-progress-20261011/ altında LINKEDIN.md,
H2_IMPACT.md, figure-data.csv, layout-check.json ve artifact-manifest.json;
iki üretim/paketleme scripti; output/ altında paylaşılabilir dosyalar.
Önceki negatif sonuç taslağı ve raporları korundu.

## Tekrar üretim

Depo: NeuroKinematics-main, önceki C1-06S henüz commitlenmemiş çalışma
ağacı ve kullanıcı değişiklikleri korundu. Bu görev commit/push yapmadı.
Belge r1; yazılım/paket sürümü değişmedi. Yeni model/checkpoint/seed yok.
Analizler mevcut JSON kanıtlarına dayanır; kaynak/output SHA'ları
artifact-manifest.json içindedir. Çıktı: 6 sayfa, 1080×1350 PNG.

Windows; grafik üretimi sistem Python 3.11/Matplotlib3.10.8; PDF ReportLab,
Pillow ve PyMuPDF için Codex bundled Python. Fontlar Windows Segoe UI.
Poppler ikinci renderer olarak kullanıldı. Bu bir yeni kinematik ölçüm
veya runtime performans karşılaştırması değildir. Süre NOT_MEASURED.

Çalıştırılan komutlar:

1. PDF becerisi marker'ı: bundled Node ile
   mark_artifact_operation_started.mjs --operation-kind create
   --expected-output-count 1 --output-format pdf (exit0).
2. python scripts/create_core_progress_media.py charts (exit0).
3. Bundled Python ile scripts/create_core_progress_media.py document (exit0).
4. Bundled Poppler pdftoppm -r 72 -png
   output/pdf/NeuroKinematics_Foundations_Core.pdf tmp/pdfs/core-progress/page
   ve pdfinfo aynı PDF (exit0).
5. Bundled Python ile scripts/package_core_progress.py (exit0).

İlk bağımlılık keşfi bundled ortamda Matplotlib olmadığını gösterdi;
sistem Python'daki mevcut3.10.8 kullanıldı. Paket kurulmadı. Yazılan grafik
scriptindeki çok satırlı string, ilk yürütmeden önce düzeltildi.

## Test ve ham kanıt

Yeni solver/test/eğitim koşusu yapılmadı. Mevcut C1-06S315 test sonucu
tarihsel kanıt olarak kullanıldı. Medya için şu kontroller yapıldı:

- C1-07 manifestindeki üç FK_TANH validation A/B0/3600 doğrulandı.
- C1-06S validation3600/3600, eksik teacher351, final12000/12000 ve
  60000 ölçüm sayıları JSON'dan okundu; grafik CSV'sine aktarıldı.
- Kapaktaki TCP izdüşümünün12000 sorgu dosyası SHA'sı doğrulandı.
- PDF6 sayfa, PNG6 adet ve1080×1350 boyut; Türkçe metin çıkarımı PASS.
- Metin sınır denetimi PASS; PyMuPDF ve Poppler render'ları görsel
  incelendi. Taşma, kesilme veya çakışma gözlenmedi. Grafik eksenleri0–100.
- Tam boyut4/5. görsellerde sayı/etiket okunabilirliği ayrıca incelendi.
- Gönderi metni2325 karakter; düz TXT, alternatif metin TXT ve6PNG ZIP hazır.

PDF: output/pdf/NeuroKinematics_Foundations_Core.pdf.
Görseller/metin/ZIP: output/linkedin/neurokinematics-core-progress/.
Önizleme: aynı klasörde overview.png. İkinci renderer kanıtı:
tmp/pdfs/core-progress/poppler-overview.png. Kaynak haritası:
docs/publication/core-progress-20261011/artifact-manifest.json.

## Sonuç ve yorum

İlk iki fazın tamamlanması başarı diliyle anlatıldı; Core araştırma
kapanışı ve yeni geometri sonucu açık ayrıldı. Neural katkının henüz
sağlanmadığı/gösterilmediği, fiziksel robot ve çarpışma sınırları korundu.
Sıfır validation'dan3600 geometri çözümüne grafik bir eğitim kazanımı
olarak sunulmadı. Final12k ile validation3,6k farklı veri setleridir.

H2'nin görev sırası/kabul eşikleri korunabilir; geometrik adayın akışı ve
adil kıyas tasarımı için uygulama eki gerekir. "Her şey stabil" sonucu
desteklenmez. Yeni maksimum103,1543ms ölçümün varlığı plan etkisi notunda;
kamusal görsellerde hız/gerçek zaman iddiası yoktur.

## Sonraki adım

Kullanıcı metin ve PDF veya PNG serisini paylaşabilir; bu görevde LinkedIn'e
gönderim yapılmadı. Yeni kod/kanıt GitHub'a henüz yüklenmediği için metinde
güncel sonuçları uzakta mevcut gösteren bir repo bağlantısı kullanılmadı.
H2-01 öncesi geometri kaynağı ve aday sözleşmesi; H2-02/03 için akış ve
kıyas eki kaydedilmeli. H2 NOT_STARTED; sayısal eşikler ve faz durumu aynı.
