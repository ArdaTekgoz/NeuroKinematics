# KR 6 R900 sixx geometri bileşeni

11 Ekim 2026 · Belge r1 · Uygulama kimliği `analytic-kr6-r1`.
Proje paket sürümü 0.1.0 değiştirilmedi. Öğrenilmiş ağırlık: yok.

Kaynak: `src/neurokinematics/solvers/analytic.py`.
Kimlik ve tam girdi SHA'ları: [final-freeze.json](final-freeze.json),
[HANDOFF.json](HANDOFF.json). Yapı ve türetim: [ADR-026](../../docs/adr/ADR-026-geometric-core-recovery.md).

## Arayüz

`AnalyticIK.solve(q_current, target_position_m, target_quaternion_wxyz)`

Altı eklem manifest sırasıyla radyan; poz metre, base_link'e göre tool0;
quaternion wxyz. FK/URDF/TCP byte hashleri kurulumda kontrol edilir.
q_current sınırlı eklem uzayında dal yakınlığını seçer. Çözücü hedef
eklemini, teacher etiketini, veri kimliğini veya split bilgisini almaz.

Sonuç: `status`, `q_candidate`, `position_m`, `orientation_rad`, üretilen/
kontrol edilen aday sayısı, `elapsed_ns`, `collision=NOT_CHECKED`.
Statüler KINEMATIC_VALID / INVALID_INPUT / UNRESOLVED / NUMERICAL_FAILURE.
KINEMATIC_VALID, bağımsız FK ile Profil B'yi ve katı limitleri geçer.
Süre bütçesi/iptal bu API'nin parçası değildir. UNRESOLVED erişilemezlik
kanıtı değildir. Girdi q veya hedef dizileri değiştirilmez.

## Geometri ve dallar

URDF sabit dönüşümlerinden bilek merkezi ayrılır; omuz/dirsek için düzlem
üçgeni ve bilek için X-Y-X ayrışımı çözülür. Limit içindeki 2π eşdeğerleri
aday olarak tutulur; gerçek sınırlı-eklem mesafesiyle sıralanır. En yakın
doğrulanmış aday döner. q5=0 tekilliğinde q4+q6 kısıtı ve ağırlıklı yakınlık
birlikte çözülür. Omuz ekseninde q1 için mevcut değer + 33 sabit açı adayı
vardır; sürekli çözüm ailesinin eksiksiz tarandığı iddia edilmez.

1e-12 rad seviyesindeki analitik uç nokta yuvarlaması kutunun içine
oturtulur ve ardından FK tekrar hesaplanır. Fiziksel eklem limitleri
genişletilmez. Yanlış TCP dönüşümü testi bağımsız kabul denetiminde reddedilir.

## Ölçülen kapsam

3.600 eski validation + 12.000 yeni final sorgusu, A/B tamamı başarılı.
Final main 10.000; boundary/singularity 1.000'er, hepsinde yarı local/wide.
Yeni final kökleri train/validation hedefleriyle exact duplicate değildir.
Eski finalden örnek okunmadı. Beş geçişin q sonuçları birebir aynı;
ayrı kaynak ağacında 12.000 sonuç/FK metriği birebir tekrarlandı.

Tek robotun analitik geometri başarısıdır; yeniden eğitilmiş model,
robotlar arası genelleme veya neural katkı ölçümü değildir. Gelecekteki
neural dal/arama politikası bu bileşenle aynı koşullarda karşılaştırılmalı.
Collision, fiziksel güvenlik, hareket sürekliliği ve servo kontrolü yoktur.
Detaylı oran ve süreler: [kurtarma kararı](RECOVERY_DECISION.md).

## Tekrar üretim

Komutlar ve immutable attempt yolları [RUN_REPORT](RUN_REPORT.md) içindedir.
Varsayılan üretim/audit komutları önceki kanıtların üzerine yazmayı reddeder.
Final kod/config değiştirildiğinde yeni araştırma kimliği gerekir.
Ham sorgu, ayrı q_target tanıkları ve sonuçlar `data/generated/C1-06S/`
altında Git dışındadır. Bu tur önceki 11 Ekim arşivi güncellenmedi.
