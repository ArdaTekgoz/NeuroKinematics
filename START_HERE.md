# NeuroKinematics — önce buradan başla

11 Ekim 2026 · Devir r1 · Ana plan r3. **Hybrid'e giriş hazır; uygulama başlamadı.**

1. [Çalışma kuralları](AGENTS.md).
2. [Mevcut durum, açık hedefler, kurulum ve checkpoint](docs/records/PRE_HYBRID_HANDOFF.md).
3. [Ana roadmap](docs/roadmaps/MASTER_ROADMAP.md) ve [H2 ek şartname](docs/roadmaps/H2_REVISION_SPEC.md).
4. [C1-07 neural devir](experiments/C1-07/HYBRID_HANDOFF.json) ve [C1-06S geometrik devir](experiments/C1-06S/HANDOFF.json).
5. [İlk uygulama görevi: H2-01](docs/tasks/H2-01.md).

G0/G1 araştırma kabulü ve C1-06S geometrik çözüm kabulü PASS'tir.
Geometrik çözümün yeni final başarısı 12.000/12.000 Profil A/B'dir; bu yeni neural
model eğitimi veya tüm robot uzayı garantisi değildir. Doğrudan neural hedef
NOT_MET; H1 neural katkı NOT_MEASURED. Yeni eğitim H2-N1/N2, katkı ölçümü H2-03'tedir.

H2-01'de ortak istek/sonuç ve motor kimliği sözleşmesi uygulanır. Ardından
H2-02 → H2-N1 → H2-N2 → H2-03 → H2-04 → H2-05 → H2-06 bağımlılık sırası izlenir.
Eski başarısız neural modeli tek başına komut üreticisi olarak kullanma.
G2 ve G-N kararlarını ayrı tut; eski final verisini yeni model seçimine açma.

Kod/plan/kanıt için Git checkpoint; büyük veri/model için ayrı artifact arşivi gerekir.
[Checkpoint kayıtları](experiments/PRE-HYBRID-20261011/) tam kapsamı ve gerçek doğrulama sonucunu taşır.
