# ADR-028 — Hybrid öncesi güncel devir ve tarihsel kanıt korunması

11 Ekim 2026 · ACCEPTED · Kapsam: belge/teslim; çözücü mimarisi ve eşikler değişmedi.

## Gerekçe

C1-07 tesliminin README/CORE_RESUME hashleri geometrik kurtarma ve roadmap r3'ten
önceki durumu gösteriyor. Güncel giriş belgelerini eski tutmak yeni geliştiriciyi
yanlış plana yönlendirir. Eski teslim hashlerini değiştirmek ise kanıtı bozar.

## Karar

Eski girişler ve revize edilen r3 teslim dosyaları aynı baytlarla
`docs/records/archive/pre-hybrid-20261011/` altında korunur. Açık original/snapshot/SHA
eşlemesi `experiments/PRE-HYBRID-20261011/historical-snapshots.json` içindedir.
Tarihsel doğrulayıcı yalnız beklenen SHA eşlemede aynıysa arşiv kopyasına yönelir;
eşleme dışındaki bütün girdiler canlı orijinal dosyada doğrulanır. Orijinal C1-07,
C1-06S ve plan manifestleri değişmez. Güncel devir için yeni kanıt üretilir.

README, AGENTS, CURRENT_ROADMAP, CORE_RESUME ve START_HERE tek güncel durum bildirir:
G0/G1 araştırma PASS; C1-06S geometrik PASS; direct neural NOT_MET; H1 NOT_MEASURED;
H2-01 NOT_STARTED. Yazılım sürümü artırılmaz; release veya tag kabulü çıkarılmaz.

Checkpoint kapsamına bu çalışma dizisindeki kod/test/plan/belge/kanıt alınır;
önceden mevcut ilgisiz kullanıcı belgeleri ve kullanıcı STATUS/TRACE değişiklikleri
ayrı korunur. Büyük araştırma girdileri, model ağırlıkları ve sealed ham kayıtlar
Git dışında SHA manifestli arşivde opak bayt olarak saklanır. Ortamlar/cache/secrets
arşiv kapsamına girmez. Arşiv doğrulama, boş hedef restore ve bundle clone gerekir.

Kullanıcının seçtiği `C:\Users\Arda TEKGÖZ\Documents\NK_Arşiv` ikinci kopya hedefidir;
aynı C: diski disk arızasına dayanıklı bağımsız aygıt olarak kabul edilmez.

## Doğrulama ve sonuçlar

[Çalışma kaydı](../../experiments/PRE-HYBRID-20261011/RUN_REPORT.md) gerçek sonuçları taşır.
Plan/hash/link denetimi, Git teslimi, artifact restore ve ikinci kopya doğrulaması
ayrı kaydedilir. Çalıştırılmamış Hybrid/neural/ONNX/collision testleri NOT_RUN kalır.
