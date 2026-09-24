# G0 karar belgesi

Karar: PASS / ACCEPTED
Tarih: 24 Eylül 2026 · Belge r1 · Hedef yazılım v0.1.0
Foundations: COMPLETE
Core: READY / NOT_STARTED

| Zorunlu gerekçe | Kanıt ve sonuç |
|---|---|
| Model identity ve koordinat sözleşmesi | Immutable robot manifesti 29 dosya; T-F01 16/16 PASS |
| FK ve Jacobian | T-F02 102/102 (10000 q), F0-03 159/159; orijinal eşikler korunuyor |
| Veri manifesti, split ve leakage | İki gerçek smoke üretimi; 6 shard eş; tüm alt kümelerde grup/q split kesişimi sıfır |
| DLS tekrar üretimi | 384 query × 2 deadline; koşu başına 768 bağımsız doğrulanmış satır |
| Kritik hata yolları | F0-05 32/32 mutasyon; F0-06 26/26 test, 25 negatif kontrol |
| Kanıt zinciri | 212 eski checksum, 18 Stage 1 kayıt; immutable dosyalar ve 14 yerel büyük dosya hashleri PASS |
| Temiz kurulum ve T-F09 | Yeni detached F0-05 worktree + hashli F0-06 overlay; locked install/lock-check, 497 regresyon testi ve gerçek iki üretim PASS |
| Devir girdileri | handoff-inputs.json; robot/config/schema/runtime/veri/benchmark kimlikleri hashli |

Karar için başarı oranı alt sınırı yoktur. F0-05 geniş başlangıç düşük başarı
oranları ölçülmüş baseline olarak korunur. Benchmark sürelerinin ham hash eşitliği
aranmaz; query/solver/profile/deadline/seed/schema/limit ve kayıt tamlığı denetlenir.

Tam komut ve ham kanıtlar [RUN_REPORT.md](RUN_REPORT.md),
[FOUNDATIONS_EVIDENCE_INDEX.json](FOUNDATIONS_EVIDENCE_INDEX.json),
[SHA256SUMS](SHA256SUMS) içinde. Kritik açık hata: YOK.

- Linux NOT_RUN; kilitte linux-64 bulunması yürütme kanıtı değildir.
- Fiziksel robot doğruluğu, kalibrasyon, collision checking ve gerçek robot güvenliği doğrulanmadı.
- F0-05/F0-06 süre ölçümleri bu Windows hostuna özgüdür; gerçek zaman garantisi yoktur.
- DLS geniş başlangıç performansı sınırlıdır; ölçülen haliyle baseline kalır. Başarı oranı eşiği türetilmedi.
- Coverage ampiriktir; matematiksel tam kapsama/erişilebilirlik garantisi değildir.
- Harici solver baseline'ları Core kapsamındadır. Eğitilmiş neural model yoktur.
- Üretici PDF raw bayt/hash eksikliği önceki görevlerden devralınmıştır.
- Foundations kapanışı üretim veya fiziksel robot güvenlik onayı değildir.
