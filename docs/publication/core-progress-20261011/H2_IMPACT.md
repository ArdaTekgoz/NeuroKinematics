# H2 planına etkisi

11 Ekim 2026 · Belge r1 · Açıklama ve önerilen uygulama eki; H2 başlatılmadı.

**Ana roadmap kullanılabilir; uygulama planı bütünüyle değişmeden kalamaz.**
G1 kabulü ve yeni C1-06S devir paketi hazırdır. Yeni bileşen eğitilmiş neural
model değil, sabit robotun analitik geometrisidir. ADR-026 bu ayrımı korur.

| Görev | Korunan amaç | Eklenmesi / netleştirilmesi gereken nokta |
|---|---|---|
| H2-01 | Girdi, kimlik, kabul, hata ve geç sonuç politikası | Geometrik adayın kimliği/kaynak türü; UNRESOLVED ve collision NOT_CHECKED; tüm adaylarda aynı son doğrulama |
| H2-02 | Toplam süre bütçesi, refinement ve sınırlı fallback | Geometrik adayın akıştaki yeri, süresi ve desteklenen robot sınırı; mevcut neural+sayısal politikaya etkisi için uygulama ADR'si |
| H2-03 | Aynı algoritma/bütçede H1 neural seed kıyası | Mevcut eşli H1 kontrolü korunmalı. Geometrik taban ayrı kontrol olarak eklenmeli; farklı motorları karşılaştırıp sonucu yalnız neural katkı diye yorumlamamalı |
| H2-04 | Zamanlı yörünge, sıçrama, çarpışma | Çoklu analitik dallar arasındaki seçimin sürekliliğe etkisi; statik IK başarısı bu testlerin yerine geçmez |
| H2-05 | Öğrenilmiş bileşende PyTorch/ONNX eşliği | ONNX'e hangi neural adayın aktarılacağı açık seçilmeli. Analitik Python çözücü neural model diye export kapsamına yazılmamalı |
| H2-06 | G2, desteklenen motor ve servis paketi | Geometrik/sayısal/neural rollerinin ölçülen sonuca göre etiketlenmesi |

Görev sırası, bağımlılıklar ve başarı eşikleri düşürülmez. Bu belge H2
roadmap dosyasını veya mevcut H1 hipotezini değiştirmedi. Ayrıntılı akış ve
kıyas konfigürasyonu H2-01/02 hazırlığında sonuç görülmeden kaydedilmelidir.

## Stabilite hakkında söyleyebildiğimiz

Yeni bileşen bu robotta 12.000 yeni hedefi aynı profil eşikleriyle çözdü;
315 test, bağımsız FK ve ayrı kaynak ağacı tekrarı geçti. Bu kapsamda
doğrulanmış bir kinematik çözüm temeli var.

Yeni finalin CPU batch1 P95 bileşen süresi 0,5029 ms, maksimumu 103,1543 ms;
60.000 ölçümün biri 50 ms'yi aştı. Uçtan uca deadline/iptal davranışı,
yörünge/çarpışma, ONNX, fiziksel robot ve neural fayda henüz doğrulanmadı.
Dolayısıyla bütün sistemin stabil veya üretime hazır olduğu söylenemez.

Kaynaklar: [H2 roadmap](../../roadmaps/H2_Hybrid.md),
[Hybrid tasarım](../../raporlar/03_Hybrid_v2_0_r1.md),
[ADR-026](../../adr/ADR-026-geometric-core-recovery.md),
[C1-06S kararı](../../../experiments/C1-06S/RECOVERY_DECISION.md).
