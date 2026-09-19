# Foundations faz roadmap

Hedef v0.1.0 · Belge r6 · Durum DEVAM EDİYOR · Etkin emek 60–90 saat

## Amaç

FK ve Jacobian doğruluğu; veri ve benchmark güvenilirliği. Ayrıntılı tasarım: [faz raporu](../raporlar/01_Foundations_v0_1_r1.md).

## Giriş şartı

Kod öncesi kapsam, model ve ortam bilgisi. Kapsam ve sayısal eşikler deneye başlamadan dondurulur.

## Uygulama sırası

| Sıra | Görev | Bağımlılık | Emek |
|---|---|---|---|
| 1 | [F0-00 Kapsam ve ortam sözleşmesi](../tasks/F0-00.md) | Yok | 5–8 saat |
| 2 | [F0-01 Robot modeli ve manifest](../tasks/F0-01.md) | F0-00 | 7–10 saat |
| 3 | [F0-02 Bağımsız ileri kinematik doğrulaması](../tasks/F0-02.md) | F0-01 | 12–18 saat |
| 4 | [F0-03 Jacobian ve metrik doğrulaması](../tasks/F0-03.md) | F0-02 | 8–12 saat |
| 5 | [F0-04 Deterministik veri fabrikası](../tasks/F0-04.md) | F0-03 | 12–18 saat |
| 6 | [F0-05 Sayısal baseline ve ölçüm altyapısı](../tasks/F0-05.md) | F0-04 | 10–14 saat |
| 7 | [F0-06 Foundations kapanış ve faz devri](../tasks/F0-06.md) | F0-05 | 6–10 saat |

## Görev bazında kabul ve değerlendirme

### F0-00 Kapsam ve ortam sözleşmesi

**Gereksinim:** REQ-F01. **Test:** T-F00.

**Çıktı:** docs/SPEC.md; ortam kilidi ve kurulum tarifi.

**Kabul:** Robot sınıfı, görev, birimler, bütçe profilleri ve kapsam dışı maddeler tanımlı; küçük Python ortam kontrolü çalışır.

**Başarısızlıkta:** Ortam kurulmazsa kurulum hatasıyla ENGELLİ kaydı aç; doğrulanmış başka ortamı ADR ile seç.

**Kanıt:** [`RUN-20260918-001`](../../experiments/F0-00/RUN_REPORT.md). Mevcut sonuç: T-F00 PASS, F0-00 TAMAMLANDI.

### F0-01 Robot modeli ve manifest

**Gereksinim:** REQ-F01. **Test:** T-F01.

**Çıktı:** assets/robots/robot_a/manifest.json; kaynak ve SHA256 kaydı.

**Kabul:** URDF varyantı, base, tip, TCP, etkin joint sırası, limitler, eksenler ve kaynak/lisans alanları tam; eksik girdi reddedilir.

**Başarısızlıkta:** Model kaynağı belirsizse geometri uydurma; kaynak edin veya robot değişikliğini ADR ile kaydet.

**Kanıt:** [`RUN-20260918-002`](../../experiments/F0-01/RUN_REPORT.md). Mevcut sonuç: T-F01 16/16 PASS, F0-01 TAMAMLANDI. Uygulama commit'i `4048c428afceaab4418d6107897dcd36c2d48f33`; immutable çıktılar F0-02'de aynı hashlerle korundu.

### F0-02 Bağımsız ileri kinematik doğrulaması

**Gereksinim:** REQ-F02. **Test:** T-F02.

**Çıktı:** kinematics servisi; FK karşılaştırma sonuçları.

**Kabul:** Float64 ve aynı modelde kapanışta 10.000 q için maksimum konum farkı ≤1e−9 m, dönme matrisi norm farkı ≤1e−9.

**Başarısızlıkta:** Frame, TCP, eklem sırası ve çarpım sırasını düzelt; eğitim açma.

**Kanıt:** [`RUN-20260918-003`](../../experiments/F0-02/RUN_REPORT.md). T-F02 102/102 PASS; F0-02 TAMAMLANDI. 10000 float64 q; PCG64 seed 20260918; maksimum konum `4.75098925995612e-16 m`, rotation Frobenius `9.159602786276758e-16`; iki eşik ≤1e-9, aşım/nonfinite sıfır. Uygulama commit'i `d92dd213bb96f8932bd0019541dd13dd7365afaf`. F0-02 kapanışı F0-03'e geçişi hazırladı; güncel F0-03 sonucu aşağıdadır.

### F0-03 Jacobian ve metrik doğrulaması

**Gereksinim:** REQ-F03. **Test:** T-F03, T-F04.

**Çıktı:** Jacobian adaptörü; metrik birim testleri.

**Kabul:** En az 100 q üzerinde merkezi farkla normalize Jacobian farkı ≤1e−5; 0/90/180 derece ve quaternion işaret örnekleri doğru.

**Başarısızlıkta:** Çerçeve ve h duyarlılığını ayır; metrik yanlışsa zor-alt-küme üretimini başlatma.

**Kanıt:** [RUN-20260919-001](../../experiments/F0-03/RUN_REPORT.md).
19 Eylül 2026: PASS / TAMAMLANDI. Uygulama 981f6143ce38574021edac7373586976cf97bdf4.
T-F03 256+21 q, üç h/üç çift geçti; ana maksimum 1.7676058530094515e-10.
T-F04 48/48; tam F0-03 159/159; 12 mutasyon yakalandı; regresyonlar PASS.
İlk TCP hash yazım hatası kullanıcı yetkisiyle netleşti; girdiler değişmedi.
F0-04'e geçiş hazır; F0-04 başlatılmadı.

### F0-04 Deterministik veri fabrikası

**Gereksinim:** REQ-F04. **Test:** T-F05, T-F06, T-F07.

**Çıktı:** data manifests; shardlar; split ve coverage raporu.

**Kabul:** Aynı config/seed aynı sayısal içerik hashini üretir; grup kesişimi sıfır; kayıtlar referans FK ile tutarlı; kapsama tanımı açık.

**Başarısızlıkta:** Sızıntı varsa dataset sürümünü geçersiz işaretle ve grupları yeniden böl; önce küçük shard kullan.

**Kanıt:** `experiments/F0-04/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### F0-05 Sayısal baseline ve ölçüm altyapısı

**Gereksinim:** REQ-F05. **Test:** T-F08.

**Çıktı:** DLS servisi; benchmark contract; query_results.jsonl.

**Kabul:** Kolay referans hedefler toleransta çözülür; bozuk/erişilemez örnekte yanlış başarı yok; timeout ve toplam süre kaydı mevcut.

**Başarısızlıkta:** Solver matematiği ile zaman ölçümünü ayrı tanıla; eksik iterasyon sayısını sıfır yazma.

**Kanıt:** `experiments/F0-05/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### F0-06 Foundations kapanış ve faz devri

**Gereksinim:** REQ-F06. **Test:** T-F09.

**Çıktı:** G0 kararı; temiz kurulum; Core devir kaydı.

**Kabul:** T-F00–09 kritik kontrolleri geçer; temiz ortamda küçük veri ve benchmark tekrar üretilir; sonraki faz girdileri hashli.

**Başarısızlıkta:** Geçmeyen temel kontrol varken G0 kapatma; yalnız belge tamamlandı diye kodu tamamlandı sayma.

**Kanıt:** `experiments/F0-06/`. Mevcut sonuç: ÖLÇÜLMEDİ.

## G0 kararı

FK ve Jacobian doğruluğu; veri ve benchmark güvenilirliği. Teknik kusurlar ile araştırma hipotezinin reddi farklı kararlardır. Faz raporundaki zorunlu testler ve destek profili karşılanmadan sürüm etiketi verilmez.

- [ ] Gereksinimler ve testler kapatıldı.
- [ ] Ham kayıtlar ve tekrar üretim tarifi mevcut.
- [ ] Hipotezler desteklendi, reddedildi veya belirsiz olarak yazıldı.
- [ ] Açık kısıtlar ve sonraki faz girdileri listelendi.

## Her çalışma oturumunda

Görev durumunu güncelle; ne yaptığını, hangi testin neden çalıştırıldığını ve ne öğrendiğini yaz. Bir sonraki oturumun tek ana işini belirle. [Faz devir şablonu](../templates/PHASE_HANDOFF.md) ve [deney şablonu](../templates/RUN_REPORT.md) kullanılacaktır.
