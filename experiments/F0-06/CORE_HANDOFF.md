# Faz kapanışı ve sonraki faza devir

Faz ve hedef sürüm: Foundations v0.1.0 → Core v1.0.0
Belge revizyonu: r1
Karar: G0 PASS / ACCEPTED
Tarih ve sorumlu: 24 Eylül 2026 · Codex; proje sahibi Arda Tekgöz

## Yapılanlar

F0-00–F0-06 COMPLETE. Immutable robot sözleşmesi, bağımsız/referans FK,
Jacobian/metrikler, deterministik veri, DLS baseline ve temiz ortam kapanışı tamam.
Core görevleri yapılmadı; eğitilmiş model/checkpoint yoktur.

## Doğrulama

T-F00–T-F08 resmi temiz regresyon: 497 PASS; T-F09 kapanış testleri 26/26 PASS.
Gerçek iki üretimde 384 veri/384 query ve koşu başına 768 benchmark satırı.
Hashler, komutlar ve JUnitler [kanıt indeksinde](FOUNDATIONS_EVIDENCE_INDEX.json).
G0 kararı [G0_DECISION.md](G0_DECISION.md).

F0-05 kapanış commit'i `e7d211f42496f803688e2a510daca97e102092dc`.
F0-06 uygulama commit'i `8e53698a414d34e96039e64a48c153e7decfb7b1`.
Foundations kapanış commit'i: bu belgeyle G0_DECISION.md'yi ekleyen commit;
`git log -1 --format=%H -- experiments/F0-06/G0_DECISION.md` ile exact SHA okunur.
Commitin kendi hash'ini içine yazan döngü oluşturulmadı. Push ve remote eşitliği
commit sonrası kullanıcıya ayrıca raporlanır.

## Araştırma değerlendirmesi

Kinematik hesap/kontrat ve yeniden üretim kabulü sağlandı. F0-05 50 ms B deadline
başarısı %68,627 idi; wide başlangıçlar sınırlıdır. F0-06 küçük örneği yeni bir
performans üstünlüğü iddiası değildir. Neural üstünlük hipotezi test edilmedi.

## Devir paketi

Normatif hashli liste [handoff-inputs.json](handoff-inputs.json): robot URDF,
robot_spec, manifest, TCP, pixi.lock; F0-04 config/schema/dataset manifest ve
train-only normalization; F0-05 config/solver config/benchmark contract/schema,
query/result manifestleri. URDF, joint sırası, metre/radyan, base_link/tool0,
wxyz ve immutable hashler korunmalıdır. Girdi değişikliği ayrı sürüm/ADR ve
ilgili tüm regresyonları gerektirir.

Veri sözleşmesi: grup temelli %70/%15/%15 split; aynı kök aile tek split;
normalizasyon main train'den. Query hedefleri eğitim örneği değildir; q_target
DLS çağrısına verilmez. Benchmark A/B geometri ve deadline başarısını ayrı
kaydeder; solver başarısızlığı erişilemezlik kanıtı değildir. Collision NOT_CHECKED.
Pinocchio mutable Data instance'ı eşzamanlı paylaşılmaz.

Üretim dataset/query/result dosyaları Git dışında, yerel 14 dosyanın hashleri
`history-audit.json` içindedir. Başka hostta var oldukları varsayılmaz; manifestteki
üretim tarifleri kullanılır. Küçük F0-06 kanonik shard/query/result dosyaları bu
pakette saklanır. İlk üretim yolunu gösteren manifest alanları tarihsel provenance'dır;
replay komutu [COMMANDS.md](COMMANDS.md) içinde açık config ile kayıtlıdır.

```powershell
python scripts/run_f06_clean.py --worktree C:/Users/Public/NeuroKinematics-F006-next --output temp/f06-next
```

Hedefler yeni olmalıdır. Kanonik platform native Windows x64; global paket
indirme cache kullanılabilir, mevcut proje ortamı/verisi kopyalanmaz.

## Sonraki faz

Core READY / NOT_STARTED. C1-01, C1-02 ve C1-03'ün ortak F0-06/G0 ön koşulu
sağlandı; üçü de NOT_STARTED. Roadmap listesi C1-01 → C1-02 → C1-03 olarak
korunur; aralarına yeni bağımlılık veya öncelik eklenmedi. C1-04 için C1-02 ve
C1-03, C1-06 için harici baseline ve model deneyleri ayrıca gereklidir.
Bu görevde hiçbir Core işi, tag veya release başlatılmadı.

Planlanan F0-06 emek 6–10 saat; etkin insan emeği ölçülmedi, fark hesaplanamaz.
Komutların gerçek UTC aralıkları records'ta; bunlar insan emek süresi değildir.

- Linux NOT_RUN; kilitte linux-64 bulunması yürütme kanıtı değildir.
- Fiziksel robot doğruluğu, kalibrasyon, collision checking ve gerçek robot güvenliği doğrulanmadı.
- F0-05/F0-06 süre ölçümleri bu Windows hostuna özgüdür; gerçek zaman garantisi yoktur.
- DLS geniş başlangıç performansı sınırlıdır; ölçülen haliyle baseline kalır. Başarı oranı eşiği türetilmedi.
- Coverage ampiriktir; matematiksel tam kapsama/erişilebilirlik garantisi değildir.
- Harici solver baseline'ları Core kapsamındadır. Eğitilmiş neural model yoktur.
- Üretici PDF raw bayt/hash eksikliği önceki görevlerden devralınmıştır.
- Foundations kapanışı üretim veya fiziksel robot güvenlik onayı değildir.
