# NeuroKinematics Foundations başlangıç değerlendirmesi

- Tarih: 17 Eylül 2026
- Durum: DEPO VE DOKÜMANTASYON HAZIRLIĞI
- Aktif faz: Foundations
- Görev durumu: F0-00-F0-06 PLANLANDI
- G0 kararı: VERİLMEDİ

Bu kayıt planlama paketini mevcut Git deposuyla ilişkilendirir ve bir sonraki teknik oturumun başlangıç noktasını tanımlar. Bu oturumda uygulama kodu yazılmadı; ortam kurulmadı; bağımlılık yüklenmedi; veri, model veya benchmark üretilmedi. Belge incelemesi hiçbir F0 görevinin tamamlandığı veya G0 kapısının geçildiği anlamına gelmez. Gerçek ilerleme durumu [STATUS](STATUS.md), gereksinim-görev-test ilişkisi [TRACEABILITY](../TRACEABILITY.md) üzerinden izlenir.

## Neuro ve NeuroKinematics ilişkisi

Neuro şemsiye addır; bugün bu şemsiye altında etkin olan ilk ve tek ürün hattı NeuroKinematics'tir. NeuroKinematics, seri manipülatörler için robot başına öğrenilmiş IK başlangıçları üretmeyi, adayları bağımsız kinematik hesaplarla doğrulamayı ve gerektiğinde sayısal iyileştirme kullanmayı hedefleyen araştırma ve mühendislik platformudur. NeuroKinSim adı ileride Studio içindeki görselleştirme bileşeninin tarihsel adı olarak kalabilir. NeuroLocalization, algılama ve tam robot kontrolü için ayrı bir geliştirme hattı bugün açılmaz. Bu ürün ve depo kararı [ana plan](../raporlar/00_Neuro_Ana_Plan_r1.md) ile [ADR-001](../adr/ADR-001-surumler.md) içinde tanımlıdır.

## Fazların amacı ve sınırı

| Faz | Amaç | Çıkış kapısı | Bu başlangıçtaki durum |
|---|---|---|---|
| Foundations v0.1.0 | Robot kimliğini, FK/Jacobian matematiğini, veri üretimini ve sayısal benchmark temelini doğrulamak | G0 | PLANLANDI; test sonucu yok |
| Core v1.0.0 | Öğrenilmiş IK modellerinin ve model bilgili kayıpların katkısını kontrollü deneylerle ölçmek | G1 | BAŞLATILMADI |
| Hybrid v2.0.0 | Nöral başlangıcı bağımsız doğrulama, bütçeli sayısal refinement, yörünge/geometri analizi ve ONNX hattıyla sınamak | G2 | BAŞLATILMADI |
| Studio v3.0.0 | Doğrulanmış servisleri ikinci robot, masaüstü iş akışı, dışa aktarım ve paketleme ile kullanıcıya sunmak | G3 | BAŞLATILMADI |

Faz ilişkileri ve kapı mantığı [MASTER_ROADMAP](../roadmaps/MASTER_ROADMAP.md) ve [ana planın sürümler bölümü](../raporlar/00_Neuro_Ana_Plan_r1.md) içinde tanımlanır. Core, Hybrid ve Studio raporları ilerideki tasarım ve bağımlılıkları açıklar; tamamlanmış yazılım veya deney sonucu değildir: [Core](../raporlar/02_Core_v1_0_r1.md), [Hybrid](../raporlar/03_Hybrid_v2_0_r1.md), [Studio](../raporlar/04_Studio_v3_0_r1.md).

## Foundations kapsamı

Foundations içinde sabit tabanlı açık seri zincir, başlangıçta altı döner eklem, base çerçevesinde TCP pozu, metre-radyan-saniye birimleri ve `w,x,y,z` quaternion API sözleşmesi ele alınır. Planlanan teknik teslimat; doğrulanmış robot manifesti, Pinocchio referansı, bağımsız sınırlı seri-zincir FK, Jacobian denetimi, deterministik ve grup sızıntısı denetimli veri fabrikası, DLS baseline, sorgu başına benchmark kaydı ve temiz ortam tekrar üretimidir. Ayrıntılı sözleşmeler [Foundations teknik raporunda](../raporlar/01_Foundations_v0_1_r1.md), görev ve kabul sırası [F0 roadmapinde](../roadmaps/F0_Foundations.md) bulunur.

Şunlar sonraki fazlara bırakılmıştır:

- Neural model eğitimi, differentiable Torch FK ve kontrollü ablasyonlar Core kapsamındadır.
- Neural seed, refinement/restart politikası, zamanlı yörünge, çarpışma sonrası denetim ve ONNX Hybrid kapsamındadır.
- İkinci robotta yeniden üretim, GUI, workspace görünümü, paketleme ve kullanıcı testi Studio kapsamındadır.
- Gerçek robot kontrolü, safety-certified davranış, MAML, GNN, RL, sim-to-real ve NeuroLocalization ana sürüm hattının mevcut kapsamı değildir; [araştırma birikimi](../RESEARCH_BACKLOG.md) ile ayrıca değerlendirilir.

Kinematik geçerlilik; çarpışmasızlık, dinamik uygulanabilirlik, fiziksel kalibrasyon veya robot güvenliği kanıtı olarak yorumlanmayacaktır. Bu iddia sınırları [CLAIMS](../CLAIMS.md) ve [ortak test protokolünde](../TEST_PROTOCOL.md) korunur.

## F0 görev bağımlılık sırası

1. [F0-00 Kapsam ve ortam sözleşmesi](../tasks/F0-00.md): `docs/SPEC.md`, ortam kararı, bağımlılık kilidi ve T-F00 kanıtı.
2. [F0-01 Robot modeli ve manifest](../tasks/F0-01.md): doğrulanmış URDF varyantı, base/tip/TCP, etkin eklem sırası, limitler, kaynak, lisans ve SHA256 kayıtları.
3. [F0-02 Bağımsız ileri kinematik doğrulaması](../tasks/F0-02.md): Pinocchio referansı ile bağımsız FK karşılaştırması.
4. [F0-03 Jacobian ve metrik doğrulaması](../tasks/F0-03.md): merkezi fark Jacobian denetimi, çerçeve/ölçek ve yönelim metriği testleri.
5. [F0-04 Deterministik veri fabrikası](../tasks/F0-04.md): içerik hashleri, shardlar, grup tabanlı split, kapsama ve veri doğruluğu raporu.
6. [F0-05 Sayısal baseline ve ölçüm altyapısı](../tasks/F0-05.md): DLS servisi, benchmark sözleşmesi ve sorgu başına ham sonuçlar.
7. [F0-06 Foundations kapanış ve faz devri](../tasks/F0-06.md): temiz kurulum tekrarı, G0 kararı ve hashli Core devir paketi.

Bu sıra bağımlılık sırasıdır. Önceki görevin kabulü ve hashli çıktıları oluşmadan sonraki görev tamamlanmış sayılmaz.

## G0 kapısının gerektirdiği kanıt

G0 yalnız belge varlığıyla değil, aşağıdaki çalıştırılmış ve izlenebilir kanıtlarla değerlendirilecektir:

- T-F00: robot sınıfı, görev, birimler, bütçe profilleri, kapsam dışı işler, doğrulanmış ortam ve küçük ortam kontrolü.
- T-F01: URDF/model varyantı, frame/TCP, joint sırası, limitler, eksenler, kaynak, lisans ve dosya hashleri tam robot manifesti.
- T-F02: aynı model ve float64 üzerinde kapanışta 10.000 konfigürasyonda referans/bağımsız FK karşılaştırması; planlanan en büyük konum ve dönme matrisi farkı eşikleri [Foundations raporundan](../raporlar/01_Foundations_v0_1_r1.md) alınır.
- T-F03 ve T-F04: en az 100 konfigürasyonda merkezi fark Jacobian karşılaştırması, perturbasyon duyarlılığı ve bilinen yönelim/birim örnekleri.
- T-F05-T-F07: aynı config/seed için aynı sayısal içerik, eğitim-doğrulama-test grup kesişiminin sıfır olması, kayıtların referans FK ile tutarlılığı ve açıklanmış kapsama yöntemi.
- T-F08: kolay referans hedeflerde tolerans, bozuk veya kanıtlı dış örneklerde yanlış başarı vermeyen DLS davranışı, timeout ve gerçek toplam süre kaydı.
- T-F09: temiz ortamda küçük veri ve benchmarkın yeniden üretimi; Core'a devredilen robot, veri, config ve sonuç kimliklerinin hashlenmesi.
- Her test için gerçek komut, ortam, ham çıktı, başarısız örnekler, karar ve commit bilgisi; kayıt biçimi [RUN_REPORT](../templates/RUN_REPORT.md) şablonundadır.

Bu eşikler şu anda ölçülmemiştir. Ayrıntılı örnek sayıları, toleranslar ve istatistik kuralları [TEST_PROTOCOL](../TEST_PROTOCOL.md) ile ilgili faz raporundan alınacaktır; başarısız bir sonucu geçirmek için değiştirilmeyecektir.

## Doğrulanmamış varsayımlar ve açık kararlar

- İlk robotun KUKA KR6 R900 sixx olduğu bir planlama varsayımıdır. Kesin varyant, güvenilir URDF/Xacro kaynağı, meshler, lisans, base/tip linkleri ve sabit TCP dönüşümü edinilip doğrulanmamıştır.
- Karakteristik uzunluk, etkin eklem sırası, joint limitleri ve robot/TCP hashleri henüz sabitlenmemiştir.
- Araştırma ortamı Linux olarak önerilmiştir; bu Windows çalışma alanında native Windows, WSL veya ayrı Linux ortamından hangisinin kullanılacağı, kesin OS sürümü ve bağımlılık kilidi kararlaştırılmamıştır. Karar [ADR-004](../adr/ADR-004-platform-ve-varsayimlar.md) ile uyumlu biçimde F0-00'da kaydedilmelidir.
- Pinocchio, Python, paket yöneticisi ve test aracı sürümleri kurulmamış ve kilitlenmemiştir. Mevcut bilgisayarın CPU/GPU/RAM özellikleri ve benchmark uygunluğu kayda alınmamıştır.
- Fiziksel robot, kalibrasyon ve ölçüm donanımı varlığı doğrulanmamıştır; bunlar G0 kinematik model-içi doğrulamasının kanıtı değildir.
- Projenin kendi kod lisansı ve ileri dağıtım politikası kararlaştırılmamıştır. Bu karar F0-00 başlangıcını engellemez, fakat ilk açık dağıtımdan önce ADR gerektirir.
- `NeuroKinematics_Model_Kullanim_Plani_r1.docx` görev başına yardımcı model seçimi önerir. Bu belge kabul ölçütlerini, roadmap bağımlılıklarını veya test kapsamını değiştiren normatif bir teknik kaynak değildir.

## Mevcut deponun planla ilişkisi ve eksikleri

Başlangıç commit'i `c8edeaa` mevcut çalışma ağacını yalnız `NeuroKinematics_10.08.2026.docx` ile bırakmıştır. Git geçmişinde önceki uygulama commit'leri görünse de güncel branch üzerinde kaynak kodu, paket tanımı, test, bağımlılık kilidi veya robot varlığı yoktur; önceki çalışmalar yeni kapılardan geçmeden geçerli Foundations çıktısı sayılmaz.

Revize paket depo köküne yerleştirilmiştir: günlük plan `docs/`, paylaşım DOCX'leri `raporlar/`, görseller `figures/`, tarihsel kaynaklar `archive/` altındadır. Kök kaynak raporu ile `archive/NeuroKinematics_10.08.2026_original.docx` aynı SHA256 değerini taşır; bu tarihsel kopyalar değiştirilmemiştir. Paket yapısı ve kaynakların rolü [README](../../README.md), koruma kaydı [SOURCE_MANIFEST](../../archive/SOURCE_MANIFEST.json) içinde açıklanır.

Beş revize raporun Markdown anlatı ve tabloları DOCX karşılıklarıyla karşılaştırılmış, teknik içerik farkı bulunmamıştır. DOCX'teki on düzenlenebilir matematik nesnesi Markdown formülleriyle aynı ifadeleri taşır. Mevcut [PLAN_REVIEW_r1](PLAN_REVIEW_r1.md) görsel belge kontrolünü kaydeder; bu kickoff yalnız depo yerleşimini ve içerik mutabakatını doğrular, algoritma sonuçlarını doğrulamaz.

## Kodlama öncesi gerçek engeller

F0-00'ın başlamasını engelleyen eksik bir plan belgesi yoktur. Buna karşılık, tekrarlanabilir teknik uygulamaya geçmeden önce F0-00 içinde şu iki temel karar somutlaştırılmalıdır:

1. Çalıştırılacak OS/ortam yolu, Python ve araç sürümleri, bağımlılık/lock yöntemi ve T-F00 ortam kontrolü.
2. F0-01'e girdi olacak robot kaynak edinme planı; exact KUKA varyantı, resmi veya güvenilir model kaynağı, lisans ve base/tip/TCP sahipliği.

İkinci madde çözülemezse F0-01 ENGELLİ kalmalı veya robot değişikliği yeni ADR ile alınmalıdır. Geometri bellekten doldurulmamalı ve varsayımsal model üzerinde FK/Jacobian kabul testi başlatılmamalıdır.

## Sonraki oturumun ilk görevi

Yeni talimatla [F0-00](../tasks/F0-00.md) başlatılmalıdır. Somut sıra:

1. `AGENTS.md`, F0-00 görev dosyası, Foundations raporunun kapsam/ortam bölümleri, TEST_PROTOCOL ve bu kickoff kaydı yeniden okunur.
2. Kullanılacak çalışma platformu ile mevcut OS/CPU/GPU/RAM/Python araç durumu kaydedilir; Linux önerisinden sapma gerekiyorsa karar ADR'ye bağlanır.
3. `docs/SPEC.md` içinde robot sınıfı, görev profili, base/TCP sözleşmesi, birimler, quaternion sırası, tolerans/bütçe profilleri ve kapsam dışı maddeler dondurulur.
4. Python proje ve test aracı seçimi, kurulum tarifi ve bağımlılık kilidi oluşturulur; küçük ortam kontrolü T-F00 olarak çalıştırılır.
5. Gerçek komutlar ve çıktılar `experiments/F0-00/` altında RUN_REPORT yapısıyla kaydedilir; F0-00, STATUS ve TRACEABILITY yalnız kanıt uygunsa birlikte güncellenir.

Beklenen çıktı; onaylanmış `docs/SPEC.md`, tekrar üretilebilir kurulum/lock dosyaları, çalıştırılmış T-F00 kanıtı ve F0-01 için açık robot-modeli girdi listesidir. F0-00 sonunda robot manifesti, FK sonucu, dataset, model eğitimi veya G0 kararı beklenmez.
