# Ana rapora uygulanan revizyon haritası

Belge r1 · 17 Eylül 2026. Bu tablo belge değişikliklerini izler; ilgili kod görevlerinin tamamlandığı anlamına gelmez. Orijinal rapor aynı baytlarla korunmuştur. Yeni raporlar eski metnin bütün uzunluğunu tekrarlamak yerine etkin kapsamı yeniden kurar.

## Bölüm bazında tam eşleme

| Ana rapor bölümü | Karar | Yeni hedef |
|---|---|---|
| 1 Özet | Dört sürüm ve mevcut kanıt durumu ile yeniden yazıldı | Ana plan 1; faz raporlarının amaç bölümleri |
| 2 Giriş ve motivasyon | Tek robot araştırma hedefi; ticari fayda hipotezi; state conditioning sınırı | Ana plan 1,4,6; Core 1 |
| 3 Teorik altyapı | Birim/frame, doğru metrikler, destek sınırı, zamanlı türev düzeltmeleri | Foundations 2,3,5; Core 3,4; Hybrid 5 |
| 4 Literatür ve rekabet | Hatalı kaynak/tablo çıkarıldı; birincil yöntem tablosu ve sınırlı iddia | Ana plan 3,4,8; SOURCES; CLAIMS |
| 5 Sistem mimarisi | Ortak servis ve modüller; reference FK ile training FK ayrıldı | Foundations 1,2; Core 3; Hybrid 2; Studio 3 |
| 6 Veri ve eğitim | Sentetik tek robot çekirdeği; pair_mode, teacher ve leakage; transfer ertelendi | Foundations 4; Core 2,5; Studio 2 |
| 7 Performans | Ham sonuç, toplam latency, bütün sorgular, ayrık alt kümeler | Foundations 5; Core 6; Hybrid 4,5; Studio 4 |
| 8 AI çekirdek | Core sadeleştirildi; jerk Hybrid ve MAML araştırmaya taşındı | Core 3–6; Hybrid 5; RESEARCH_BACKLOG |
| 9 GUI ve ürün | Studio kapsamına taşındı; collision Hybrid servisine bağlandı | Hybrid 5,6; Studio 3–5 |
| 10 Teknik roadmap | Eski 13 faz dört sürüm ve 25 izlenebilir göreve eşlendi | roadmaps; TRACEABILITY; tasks |
| 11 Risk | Ölçülebilir tetikleyici, negatif sonuç ve yazılım kusuru ayrımı | Ana plan 7; faz sonu risk/kararlar; görevler |
| 12 Gelecek vizyonu | Giriş şartı ve durdurma kriterli ayrı araştırma birikimi | RESEARCH_BACKLOG |
| 13 Sonuç | Tamamlanmış ürün izlenimi yerine sürüm hedefi ve kanıt düzeni | Ana plan 1,5,7; STATUS |
| 14 Kaynakça | Kaynakça seçili doğrulanmış birincil kayıtlarla yeniden kuruldu | Ana plan/faz kaynakları; SOURCES |

## Teknik revizyon kayıtları

| ID | Eski yer | Sorun veya kapsam | İşlem | Uygulanan karşılık | Uygulama görevi |
|---|---|---|---|---|---|
| R01 | 1,2,5,10 | Tek raporda bütün ürün hedefleri | BÖL | Foundations/Core/Hybrid/Studio ve araştırma birikimi | F0-00 |
| R02 | 3.4 ve 4.1 | Pieper koşulları gerekli koşul gibi | DÜZELT | Özel geometriler için yeterli yapı; genel imkânsızlık iddiası kaldırıldı | F0-00 |
| R03 | 3.4.1 ve 3.6 | Bütün analitik çözücüler tüm çözümleri verir; her 6 DoF için 8 çözüm | DÜZELT | Çözüm sayısı ve hızın geometri/algoritma/kısıta bağlı olduğu sınırı | F0-00 |
| R04 | 3.4.2 | Pseudoinverse kapalı biçiminde rank koşulu belirsiz | DÜZELT | SVD veya lineer sistem, çerçeve ve rank farkındalığı | F0-03,F0-05 |
| R05 | 3.5,8.3 | Yoshikawa ve 7+ DoF konusunda determinant yorumu | DÜZELT | Görev boyutu, ölçekli J, sigma_min, kappa ve normalize çarpım | F0-03 |
| R06 | 3.7,5.3 | Geometrik Jacobian çerçevesi açık değil | EKLE | TCP noktasında base eksenleri ve lineer/açısal satır sırası | F0-03 |
| R07 | 3.9,4.6,8.2 | Conditioning ile kesin dal/süreklilik çözümü | DARALT | Belirsizliği azaltma hipotezi; garanti kaldırıldı | C1-02,C1-04 |
| R08 | 3.11,5.6,8.7 | Tekillik, smoothness ve jerk tek kayıpta | BÖL | Core FK/limit; tekillik opsiyonel; zamanlı jerk Hybrid | C1-05,H2-04 |
| R09 | 3.12 ve 6.3 | Quaternion kararlı ve sürekli iddiası | DÜZELT | İşaret simetrisi; 180 derece; quaternion/6D ablation | C1-05 |
| R10 | 3.14,7.2,8.1 | O(1), sub-ms ve real-time ifadeleri | DARALT | Sabit graph; donanım/bütçe ve toplam süre ölçümü | F0-05,H2-03 |
| R11 | 4.2 ve 4.9 | TRAC-IK Newton-Euler/DLS açıklaması | DÜZELT | Newton temelli ve SQP bileşenleri; resmi kaynak | C1-01 |
| R12 | 4.9 ve 14 [9] | Bensadoun CVPR ve one-to-many eksikliği | DÜZELT | ICML 2022, PMLR 162:1787–1797; çoklu dağılım yaklaşımı | C1-01 |
| R13 | 14[10]–[13] | Doğrulanamayan yazar/başlık kayıtları | ÇIKAR | Yeni bilimsel dayanaklardan çıkar; SOURCE kaydıyla karantinada tut | F0-00 |
| R14 | 4.10,4.11 | Eşsiz, dakikada eğitim, kolaylık puanları | ÇIKAR | Ölçümsüz rekabet puanları yerine H1,H2,H3 ve kullanıcı işi | C1-06,S3-05 |
| R15 | 5.3,5.6,11.2.1 | Pinocchio reference ile training differentiability karışıklığı | AYIR | Pinocchio oracle ve Torch FK gradyan testleri | C1-03 |
| R16 | 6.1,10.11 | Sentetik/public/real bütün fazların zorunluluğu | TAŞI | Core sentetik; public veri ancak frame/lisans uyumu; gerçek veri araştırma | F0-04 |
| R17 | 6.2,10.3 | Joint sampling ile workspace coverage eşitlenmesi | DÜZELT | Ampirik voxel/yönelim ve doygunluk; evrensel yüzde 95 iddiası yok | F0-04 |
| R18 | 6.4 | Pair ve yakın örnek leakage kuralı eksik | EKLE | Kök grup split; q_current üretim kuralı; teacher yanlılığı | C1-02 |
| R19 | 6.5,10.12 | İkinci robot ve unseen/zero-shot aynı kapsamda | AYIR | Studio per-robot yeniden eğitim; transfer ayrı task distribution | S3-01 |
| R20 | 6.6 | Ablation beklenen çıktıları kesin başarı gibi | DÜZELT | Kontrollü deney; reddedilen veya belirsiz sonuç mümkün | C1-05,C1-06 |
| R21 | 6.7,7.4,8.5 | Jerk denklemi zaman ölçeği ve mekanik ömür iddiası | DÜZELT | Delta_t küpü; aynı zamanlama; mekanik ömür kanıtı yok | H2-04 |
| R22 | 7.8 | Tekrar birimi ve istatistik seçimi belirsiz | EKLE | 3 eğitim seed; eşleşmiş query/group; bootstrap; birincil hipotez | C1-06,H2-03 |
| R23 | 7.10 ile 8 arası P0659–P0665 | Konuşma/LLM değerlendirme artığı | ÇIKAR | Yeni raporlarda bulunmuyor; ana arşiv korunuyor | F0-00 |
| R24 | 8.6,11.9 | Refinement güvence ve fallback garantisi | DARALT | Sınırlı deneme; bağımsız son kontrol; açık başarısızlık | H2-01,H2-02 |
| R25 | 11.2.3 | Clamping sonrası doğrudan uygunluk | DÜZELT | Projection seed olarak; FK/limit/geometri yeniden kontrol | H2-02 |
| R26 | 7.6,9.2 | Tek voxel veya skorla tüm alan uygunluğu | DÜZELT | Yönelim, örnek sayısı, VERİ YOK ve ayrı metrik katmanları | S3-03 |
| R27 | 9.3,11.6.1 | Tek poz ile yol güvenilirliği karışıyor | AYIR | Post-check ve örneklenmiş yol; continuous garanti yok | H2-04 |
| R28 | 9.5,10.10 | ONNX tüm motoru taşıyormuş izlenimi | DÜZELT | Neural tensor graph export; host doğrulama/refinement ayrı | H2-05 |
| R29 | 9.6 | Her platformda standalone çalışma hedefi | DARALT | Linux araştırma; Windows CPU ilk Studio hedefi; temiz OS kapısı | S3-04 |
| R30 | 9.7 | Bulut ve SaaS genişlemesi | TAŞI | Offline temel; talep/bütçe ile araştırma birikimi | S3-04 |
| R31 | 10.15 ve 11.10 | Faz numarası/gate anlamı uyumsuz | BİRLEŞTİR | Tek G0–G3 ve gereksinim/görev/test kimlikleri | F0-06,C1-07,H2-06,S3-05 |
| R32 | 10.16 | 6–9 ay ile 35–54 hafta iş yükü | DEĞİŞTİR | 330–520 saat; yüzde 25 pay; kapasiteye bağlı takvim | F0-00 |
| R33 | 10.11 | Noise/HIL hazırlığı gerçek robot testine alternatif | AYIR | Hazırlık testi gerçek sim-to-real kanıtı sayılmaz | H2-06 |
| R34 | 11.8.3 | Lisans envanteri genel ifade | SOMUTLAŞTIR | Sürüm/kaynak/dağıtılan varlık/bildirim ve SBOM alanları | F0-01,S3-04 |
| R35 | 12 | MAML,GNN,RL,dynamics ve yeni markalar | TAŞI | Giriş/test/durdurma şartlı RESEARCH_BACKLOG | S3-05 |
| R36 | Revize inceleme fallback oranı | Refinement ile yeniden başlatma aynı ölçü | DÜZELT | r_refine ve r_restart ayrı tanımlar | H2-03 |
| R37 | Revize inceleme güncel makaleler | IKDiffuser 2025 başlığı | GÜNCELLE | 2026 v4 başlığı, kapsamı ve ön baskı statüsü | C1-06 |
| R38 | Bütün bölümler | Implemented/planned ayrımı eksik | EKLE | STATUS ve bütün testlerde ÖLÇÜLMEDİ başlangıcı | F0-00 |
| R39 | Bütün bölümler | Erişilemezlik ve geçersizlik sınıfları eksik | EKLE | PROVEN_UNREACHABLE, NOT_CONVERGED, NOT_CHECKED, LATE_VALID | H2-01 |
| R40 | Bütün bölümler | Her adımda kalıcı kayıt gereği | EKLE | 25 görev dosyası; deney, ADR ve faz devri şablonları | F0-06,C1-07,H2-06,S3-05 |

## Eski geliştirme görevlerinin yeni karşılıkları

| Eski görev grubu | Yeni görevler |
|---|---|
| A-0 Spesifikasyon | F0-00, F0-01 |
| A-1 Kinematik | F0-02, F0-03 |
| A-2 Veri | F0-04, C1-02 |
| A-3 Baseline | F0-05, C1-01, C1-04 |
| A-4 Core | C1-03, C1-05; jerk H2-04 |
| A-5 Ablasyon ve genelleme | C1-06; ikinci robot S3-01; transfer araştırma |
| A-6 Hybrid | H2-01, H2-02 |
| A-7 Benchmark | C1-06, H2-03; her fazda regresyon |
| A-8 GUI | S3-02, S3-03; collision H2-04 |
| A-9 Deployment | H2-05, S3-04; TensorRT araştırma |
| A-10 Gerçek robot | Araştırma birikimi; noise hazırlığı ayrı |
| A-11 Multi-robot ve MAML | S3-01; MAML/zero-shot araştırma |
| A-12 Ürün ve akademik çıktı | C1-07, H2-06, S3-05 |

## Denetim notu

Kaynak raporda 14 ana bölüm ve 311 OMML matematik öğesi incelendi. Bazı matematik öğeleri tek semboldür; bu sayı 311 bağımsız denklem olduğu anlamına gelmez. Yeni raporlar kendi denklem ve kapsam sözleşmelerini kullanır. Önceki revize raporun puanları deney verisi olmadığı için yeni proje performans sonucu olarak taşınmamıştır.
