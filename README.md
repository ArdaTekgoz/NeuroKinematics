# Neuro revize rapor ve geliştirme paketi

17 Eylül 2026 · Belge revizyonu r1

Bu paket, orijinal NeuroKinematics raporunu koruyarak geliştirme kapsamını dört hedef sürüme ayırır. İçindeki raporlar ve yol haritaları uygulama planıdır; yazılım ve deney sonuçları henüz doğrulanmamıştır.

Aktif geliştirme fazı Foundations'tır. Depoda çalışmaya başlamadan önce [çalışma kurallarını](AGENTS.md) ve [Foundations başlangıç değerlendirmesini](docs/records/FOUNDATIONS_KICKOFF.md) okuyun. Görev bazlı model seçimi için [ek rehber](NeuroKinematics_Model_Kullanim_Plani_r1.docx) bulunur; bu rehber roadmap veya kabul ölçütlerinin yerine geçmez.

## Okuma sırası

1. [Neuro ana plan](docs/raporlar/00_Neuro_Ana_Plan_r1.md).
2. [Uygulanan revizyonlar](docs/REVISION_MAP.md) ve [iddia kaydı](docs/CLAIMS.md).
3. [Ana roadmap](docs/roadmaps/MASTER_ROADMAP.md).
4. [Foundations raporu](docs/raporlar/01_Foundations_v0_1_r1.md) ve [F0-00 görevi](docs/tasks/F0-00.md).
5. Gerektikçe Core, Hybrid ve Studio raporları.

## Dosyalar nasıl kullanılacak

`raporlar/` içindeki beş DOCX, gözden geçirme ve paylaşım nüshasıdır. `docs/raporlar/` aynı raporların Markdown metnini içerir. Günlük geliştirme, `docs/tasks/` altındaki 25 görev ve `docs/roadmaps/` üzerinden izlenir. Değişiklik gerekçeleri ADR, deney sonuçları records ve experiments kayıtlarıyla bağlanır.

DOCX üzerinde yapılan yeni içerik değişikliği Markdown kaynağa da aktarılmalıdır; çelişkili iki plan sürdürülmez. Faz kapanışında raporun yeni belge revizyonu üretilir. Eski dosya sürümü saklanır. Şablonlar her adımda ne yapıldığını, testini, değerlendirmesini ve sonraki faza etkisini kaydetmek içindir.

## Özgün kaynaklar

`archive/NeuroKinematics_10.08.2026_original.docx` ekli ana raporun aynı baytlı arşiv kopyasıdır. Orijinal dosya değiştirilmemiştir. Ekli revize incelemenin özgün veri kopyası, tam metni ve dosya hashleri `archive/` içindedir. Eski konuşmadan gelen sandbox bağlantıları yeni paketin dosyaları sayılmaz.

## Mevcut durum

Raporlama/planlama hazırlanmıştır. Kod kurulumu, model eğitimi, benchmark, kullanıcı testi, GitHub değişikliği ve fiziksel robot işlemi bu paket içinde yapılmış sayılmaz. [STATUS](docs/records/STATUS.md) gerçek ilerlemenin başlangıç kaydıdır. İlk teknik iş F0-00'dır.

Belge hazırlığı ve kontrolleri [PLAN_REVIEW_r1](docs/records/PLAN_REVIEW_r1.md) kaydındadır. Bağımlılık ve robot varlığı envanteri için [şablon](docs/templates/DEPENDENCY_REGISTER.md) kullanılabilir.
