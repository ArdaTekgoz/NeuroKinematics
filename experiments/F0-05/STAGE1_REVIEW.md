# F0-05 Aşama 1 inceleme ve çalışma kaydı

Kimlik: RUN-20260923-F005-STAGE1
Durum: **AWAITING_USER_APPROVAL**
Görev ve gereksinim: F0-05 · REQ-F05 · A1.1–A1.8; T-F08 hazırlığı
Tarih ve sorumlu: 23 Eylül 2026 · Codex; proje sahibi Arda Tekgöz
Yazılım hedefi: v0.1.0 · Belge revizyonu: r1

Bu kayıt `docs/templates/RUN_REPORT.md` yapısını kullanır. Nihai F0-05
`RUN_REPORT.md` veya kapanış kabul raporu değildir. F0-05 TAMAMLANDI değildir.

## Soru ve değişiklik

Sabit damping kullanan DLS matematiği, bağımsız pose doğrulaması, durum modeli
ve gelecekteki benchmark sözleşmesi sonuç görülmeden sabitlenebilir mi?
Çözücü, sözleşme/şema ve küçük analitik testler eklendi. Önceki FK, Jacobian,
veri fabrikası, model, kaynak raporlar, `pixi.toml` ve `pixi.lock` değiştirilmedi.
Mimari mevcut kinematics → solvers → benchmark bağımlılık yönünü koruyor;
yeni bağımlılık veya kapsam değişikliği yok, ADR gerekmedi.

Kullanıcının 23 Eylül mesajı **“Aşama 1'e devam etmeni onaylıyorum”**, başlangıçta
görülen `~$uroKinematics_Model_Kullanim_Plani_r1.docx` dosyasını kapsam dışında
bırakarak yalnız Aşama 1'e devam etme yetkisidir. Dosyaya dokunulmadı.
**Aşama 2 onayı verilmedi.**

### Matematik ve solver parametreleri

- `e = [p_target-p_current; log(R_target @ R_current.T)]`; base eksenleri,
  TCP noktası, linear/angular sırası. F0-03 `log_so3` yeniden kullanılır.
- `S = diag(1/ell,1/ell,1/ell,1,1,1)` hem hataya hem Jacobian'a uygulanır.
- `delta = J_tilde.T @ np.linalg.solve(J_tilde @ J_tilde.T + lambda**2 I, e_tilde)`.
  DLS çekirdeğinde açık inverse yok; eski Pinocchio FK adaptöründeki base
  dönüşüm inverse'i F0-02'nin mevcut uygulamasıdır ve değiştirilmemiştir.
- `ell=0.9015 m`, `lambda=0.05`, `float64`, en çok 200 tamamlanmış update.
- Delta bileşenleri ayrı ayrı `[-0.20,+0.20] rad` aralığına kırpılır; ardından
  `q+delta` inclusive joint limitlerine projekte edilir. Limit dışı ilk q reddedilir.
- İç durma Profile B: konum ≤0.001 m **ve** yönelim ≤0.5°.
  Profile A: ≤0.002 m **ve** ≤1°. Sıfır başlangıç hatası 0 update ile başarıdır.
- A/B ilk erişim iterasyonları ayrı tutulur. Bunlar custom-FK loop tanılarıdır;
  son geometrik başarı tekrar hesaplanan bağımsız Pinocchio FK'dan gelir.
- Art arda 5 projekte adımın infinity normu ≤1e-12 rad ise ve B sağlanmıyorsa
  `STALLED`. Adaptive damping, step acceptance, LM, restart veya tuning yok.
- `solve()` yalnız başlangıç eklemleri, hedef konum/quaternion ve deadline alır.
  `q_target` API'ye verilmez; farklı q ile aynı pose başarı sayılır.

### Query sözleşmesi (üretim henüz yapılmadı)

`config.json` ve `benchmark-contract.json` tam, makine-okunur tanımdır:

| Alan | Dondurulmuş değer |
|---|---|
| RNG / dtype | PCG64 / float64 |
| Seedler | main 20260925; local 20260926; wide 20260927; boundary 20260928; singularity 20260929 |
| Main / boundary / singularity | 10000 / 1000 / 1000 benzersiz kök q |
| Main örnekleme | Joint box içinde ardışık uniform adaylar |
| Boundary | Strict normalize joint-limit uzaklığı `<0.02`; kabul sıra numarasına göre joint/taraf döngüsü |
| Singularity | Normalize sigma_min ≤0.00727741160967353; mevcut F0-04 main-train yüzde 5 eşiği |
| Local | Her eklem ±0.05 rad uniform; limit dışı/eşit q için rejection, clamp yok |
| Wide | Joint box uniform; normalize RMS uzaklık ≥0.25 |
| Dağılım | Her subset içinde even index local, odd wide: 5000/5000; 500/500; 500/500 |
| Aday üst sınırı | main/boundary 200000; singularity 100000; başlangıç başına 10000 |
| Bağımsızlık | Tüm F0-04 split/hard shardları ve kabul edilmiş F0-05 q'larıyla tam tekrar yasak; ID/group kesişimi sıfır şartı |
| Hash biçimi | q: little-endian float64, signed zero kanonik; liste: sıralı compact JSONL UTF-8/LF ve SHA-256 |
| Frame / TCP / quaternion / joint | base_link / tool0 / wxyz / joint_1…joint_6 |
| Deadline / tekrar | Aynı listeyle 10 ve 50 ms, beş measurement pass |

F0-04 train q'ları hedef olarak kullanılmaz; yalnız exclusion/audit girdisidir.
Yeniden örneklemede RNG geri sarılmaz; aday ve red nedenlerinin sayıları tutulur.
12.000 bağımsız hedef, 120.000 ölçüm satırı planlanır. Beş tekrar N'yi artırmaz.
Her deadline bloğunda ilk 20 query bir kere ayrı warm-up; ölçüm satırları warm.
Cold model hazırlığı, yükleme, warm-up ve serialization/IO ayrı raporlanacaktır.

### Durum ve ölçüm modeli

Kapalı enum: `SUCCESS`, `MAX_ITERATIONS`, `TIMEOUT`, `STALLED`,
`NUMERICAL_FAILURE`, `INVALID_INPUT`, `UNRESOLVED`, `PROVEN_UNREACHABLE`.
Guard'lı geçişler `benchmark-contract.json/status_model/transitions` içindedir.

`SUCCESS` solver'ın geçici durma bayrağıdır. Bağımsız B doğrulaması başarısızsa
doğrulama durumu `UNRESOLVED` olur; teknik solver kodu ayrı korunur. NaN/Inf,
hatalı backend şekli ve lineer solve hatası `NUMERICAL_FAILURE`; bozuk giriş
`INVALID_INPUT` üretir. Yakınsamama erişilemezlik kanıtı sayılmaz.

`PROVEN_UNREACHABLE` yalnız bağımsız analitik sertifika ile kullanılabilir.
Bu aşamadaki validator sertifika doğrulayıcısı içermez; böyle bir iddiayı
reddeder. Aşama 2'de uygulanacak dış sınır kanıtı sözleşmede tanımlıdır:
base–TCP zincirinin origin translation normları toplamı, profil toleransı ve
sayısal pay. **0.9015 m karakteristik uzunluk dış erişim sınırı değildir.**

Deadline her iterasyonda, pose değerlendirmesi ve update sonrasında kontrol
edilir. Timeout'tan sonra iterasyon sürmez; son adayın bağımsız geometrisi
incelenebilir. `total_elapsed_ns = solve_elapsed_ns + validation_elapsed_ns`.
Deadline PASS için geometri, toplam bütçe ve uygun teknik status birlikte
gereklidir; geç solver veya validator sonucu PASS değildir. Sahte saatli küçük
testler bu kuralları sınar; süre dağılımı veya 10/50 ms performansı ölçülmedi.

Collision her zaman `NOT_CHECKED`. Eksik iterasyon `null/NOT_AVAILABLE`,
ulaşılmamış ilk-hit `null`; bunlar 0 yapılmaz. NaN/Inf ve bilinmeyen enum/alanlar
reddedilir. `benchmark-schema.json` JSON Schema 2020-12 belgesidir; yerel
validator yalnız kullanılan keyword altkümesini, ardından alanlar arası
kuralları ve bağımsız FK'yı doğrular. Genel JSON Schema motoru olduğu iddia edilmez.

## Tekrar üretim

Başlangıçta `git fetch --prune origin` başarılıydı; branch `main`, HEAD ve
origin/main `84adc24a03f619d7079e4e3900b8f51ab2591ef5`.
F0-04 uygulama commit'i `16010d518c24400f6c6d43a2459456dd822f34a8`.
Mevcut çalışma commitlenmedi. Önceden izlenen dosyalarda değişiklik yok.

| Frozen dosya | İlk test öncesi ve son test sonrası aynı SHA-256 |
|---|---|
| config.json | `4330d5e7e04ca3d266e9de98426563d3675217758e797fd0b96aaf9ca0208e82` |
| solver-config.json | `20847ea703c7aee625bd809566b959add9f14fd920bf886aa5bc997442b432d5` |
| benchmark-contract.json | `b94fd64885c93439f02ea2c75d71f3802aef8b1ff83bb81631c55dbfb0f739e4` |
| benchmark-schema.json | `2e505e03b196cc4373a5f77d53ab62db51937946da2b1b15e7e33a438dd27bc1` |

`stage1-frozen-hashes.json` ilk testten önce yazıldı. `stage1-files.sha256`
son kaynak, test, inceleme ve kanıt dosyalarının bayt kimliklerini saklar;
Aşama 2 preflight'ında bu paket doğrulanmalıdır. Hash listesi kendisini içermez.

`stage1-preflight.json` robot/URDF/RobotSpec/manifest/TCP, F0-04 config/schema,
manifest dosyası ve lock gerçek hashlerini içerir. F0-03 örnek içerik hash'i
`678eb4286863026880792ef0cc3c0a9d4f92e16f85b1aa009705cbf0b59b26e7` yeniden hesaplandı.
F0-04 dataset content
`5cb4e64580ecaf99afd11b3c8b98e06ed00c712e83bf2d9ee4d8c3acd58173fe`;
mevcut 12 shard doğrulandı, yeniden üretim gerekmedi.
Lock: `56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a`.

Ortam Windows native; Python/NumPy/Pinocchio/CPU kimliği preflight JSON'da.
RAM ve gerçek thread sayısı NOT_MEASURED; GPU NOT_USED; Linux NOT_RUN.
İşin toplam etkin emek süresi NOT_MEASURED. Son test zamanı JUnit içinde
`2026-09-23T19:53:16.791783+03:00`, suite süresi 1.003 s.

Çalıştırılan temel komutlar:

```powershell
git fetch --prune origin
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
rg --files -g AGENTS.md
pixi run --locked python -m pytest -q tests/f0_05 --junitxml=experiments/F0-05/stage1-junit.xml -o junit_family=legacy 2>&1 | Tee-Object -FilePath experiments/F0-05/stage1-tests.log
git diff --check
git status --short
```

Girdiler `Get-Content` ile okundu. Hash/schema hazırlama ve read-only preflight
için PowerShell here-string içindeki Python `pixi run --locked python -` ile
çalıştırıldı. Preflight'ta `load_robot()`, F0-03 `sample_configurations` +
`sample_hash`, F0-04 `verify_dataset(Path('data/generated/F0-04/run-a'),
Path('experiments/F0-04/dataset-manifest.json'))` kullanıldı. Bu işlem yeni
benchmark hedefi veya sonuç JSONL üretmez. Normal Git diff'in untracked
dosyaları kapsamaması nedeniyle yeni dosyalar ayrıca
`git -c core.autocrlf=false diff --no-index --check -- NUL <path>` ile kontrol
edildi. Normal `git diff --check` ve 18 yeni dosyanın whitespace denetimi
geçti; checksum listesi bu 18 dosyanın ardından üretildi.

Kontrol betiğinin ilk denemesi, no-index'in normal fark anlamındaki exit 1
kodunu hata yorumladı. Düzeltilen kontrolde PowerShell logunun CRLF satır
sonları görüldü; log içeriği korunarak LF'ye normalize edildi. Son denetim
tanısız geçti. Bunlar test/matematik başarısızlığı veya parametre değişikliği
değildir; yerel Git ayarı kalıcı olarak değiştirilmedi.

## Test ve ham kanıt

Yalnız `tests/f0_05` çalıştırıldı. İlk koşu 119/119 geçti. Kod incelemesinde
giriş durumu/model kimliği kontrolleri sıkılaştırıldı; altı küçük sınır testi
eklendi. İkinci ve son koşu **125/125 geçti**, failure/error/skip sıfır.
Dört frozen JSON veya sayısal parametre bu süreçte değiştirilmedi.

| Gereksinim | Değişiklik | Test | Son kanıt |
|---|---|---|---|
| A1.1 frozen değerler | dört JSON + hash listesi | config/type/hash bozma reddi | stage1-junit.xml; stage1-frozen-hashes.json |
| A1.2 DLS matematiği | solvers/dls.py | sıfır hata; 1/2 joint; sign/frame/order; scale/damping/solve | 61 matematik/durum/smoke testi |
| A1.2 sınırlar/hatalar | projected update, hata durumları | 0.20 cap; joint projection; singular J; NaN/Inf/shape/solve exception | stage1-junit.xml |
| A1.2 bağımsız başarı | benchmark/validation.py | 6 local robot hedefi; eş pose/farklı q; sahte success; bağımsız FK arızası | stage1-tests.log |
| A1.3 durum/zaman | enum, geçişler, deadline verdict | cap/stall/timeout; geç doğrulama; null iteration; unreachable flag reddi | stage1-junit.xml |
| A1.4 query planı | config + benchmark-contract | sayılar/seedler/profiller/record count sözleşme tutarlılığı | 50 contract testi içinde; tam audit NOT_RUN |
| A1.5 şema | JSON Schema + strict parser + bağımsız validator | enum/shape/null/nonfinite/extra field/hash/metric/deadline tutarlılığı | 50 contract testi |
| A1.6 mutasyonlar | runtime production monkeypatch | aynı kabul assert'i önce geçer, mutasyondan sonra başarısız olur | 14/14 tespit; JUnit properties |

14 üretim mutasyonu: orientation sign, linear/angular sıra, local-frame log,
characteristic length kaldırma, damping sıfırlama/değiştirme, explicit inverse,
step cap kaldırma, limitleri yok sayma, yalnız position/yalnız orientation
başarısı, solver flag'ine kör güven, timeout PASS, geç sonucu PASS sayma.
Mutasyonlar diskteki üretim kaynaklarını değiştirmez. Sonuç yalnız iki dizinin
farklı olmasına değil, aynı kabul assertion'ının başarısız olmasına dayanır.

Yeni dosyaların tam listesi (önceden izlenen dosya değiştirilmedi):

```text
src/neurokinematics/solvers/__init__.py
src/neurokinematics/solvers/dls.py
src/neurokinematics/benchmark/__init__.py
src/neurokinematics/benchmark/contract.py
src/neurokinematics/benchmark/validation.py
tests/f0_05/conftest.py
tests/f0_05/test_math.py
tests/f0_05/test_contract.py
tests/f0_05/test_mutations.py
experiments/F0-05/config.json
experiments/F0-05/solver-config.json
experiments/F0-05/benchmark-contract.json
experiments/F0-05/benchmark-schema.json
experiments/F0-05/stage1-frozen-hashes.json
experiments/F0-05/stage1-junit.xml
experiments/F0-05/stage1-tests.log
experiments/F0-05/stage1-preflight.json
experiments/F0-05/STAGE1_REVIEW.md
experiments/F0-05/stage1-files.sha256
```

Son `git status --short` dört yeni dizini ve kapsam dışı Word dosyasını gösterir:

```text
?? experiments/F0-05/
?? src/neurokinematics/benchmark/
?? src/neurokinematics/solvers/
?? tests/f0_05/
?? ~$uroKinematics_Model_Kullanim_Plani_r1.docx
```

## Sonuç ve yorum

Aşama 1 matematik ve contract testleri geçti; inceleme durumu
**AWAITING_USER_APPROVAL**. F0-05 nihai kabulü verilmedi.

Açık riskler ve yapılmayanlar:

- Büyük orientation hatalarında sabit damping'li yerel DLS için global yakınsama
  garantisi yoktur. Geniş başlangıçların başarı oranı henüz bilinmiyor.
- Tam query üretimi, aday cap uygulanabilirliği, q/group duplicate audit'i ve
  ikinci üretim deterministik hash kontrolü NOT_RUN; kuralları donduruldu.
- 10/50 ms benchmark, beş ölçüm geçişi, latency/hata/iteration dağılımları,
  CPU/thread/RAM ölçümü NOT_RUN. Test süresi benchmark latency'si değildir.
- Analitik dış hedef sertifika doğrulayıcısı, query/JSONL stream doğrulama,
  aggregator, CLI, Pixi görevleri, tam T-F08, uçtan uca Aşama 2 mutasyonları,
  tam acceptance runner ve F0-00–F0-04 regresyonları bu çalışmada NOT_RUN /
  henüz uygulanmadı. Önceki test sonuçları yeni koşu gibi gösterilmez.
- Timed çalışmalarda süre yanında deadline'ın etkilediği iterasyon/status/q
  alanları da değişebilir. Byte determinizmi yalnız query/config için; küçük
  deadline'sız solver örneklerinde matematik determinizmi denetlenebilir.
- Model içi kinematik sonuçlar fiziksel doğruluk veya robot güvenliği değildir;
  collision NOT_CHECKED. Linux denenmedi.

## Sonraki adım

Kullanıcı bu paket üzerinde açık Aşama 2 onayı verene kadar dur.
Onay sonrasında çalışma ağacı ve `stage1-files.sha256` karşılaştırılmalı;
onay ve aynı kalan frozen hashler kaydedilmeli; Aşama 1 testleri yeniden
çalıştırılmalıdır. Matematik, solver parametreleri veya benchmark sözleşmesi
değişecekse yeniden açık onay gerekir.

CLI, tam query manifesti/listesi, `query_results.jsonl`, benchmark özetleri,
nihai RUN_REPORT, kapanış raporu, commit veya push bu aşamada oluşturulmadı.
STATUS/TRACEABILITY/roadmap tamamlandı yapılmadı; F0-06 başlamadı; G0 açık.

## 24 Eylül 2026 — Aşama 2 onay kaydı

Kullanıcı daha sonra **“Aşama 2'ye geçmeni açıkça onaylıyorum”** mesajını verdi.
Onay `stage2-approval.json` içinde özgün metniyle kaydedildi; bu bölüm Aşama 1
incelemesinin tarihsel `AWAITING_USER_APPROVAL` durumunu değiştirmez. Onaylanan
paketin SHA-256 değeri `05884b13817648dbfb0a4c3780905dc25d3f488bd11cf5379ecfbd92f4feac61`.
Onay sonrası aynı kalan dört frozen girdi:

| Dosya | SHA-256 |
|---|---|
| config.json | `4330d5e7e04ca3d266e9de98426563d3675217758e797fd0b96aaf9ca0208e82` |
| solver-config.json | `20847ea703c7aee625bd809566b959add9f14fd920bf886aa5bc997442b432d5` |
| benchmark-contract.json | `b94fd64885c93439f02ea2c75d71f3802aef8b1ff83bb81631c55dbfb0f739e4` |
| benchmark-schema.json | `2e505e03b196cc4373a5f77d53ab62db51937946da2b1b15e7e33a438dd27bc1` |

Aşama 1 testleri onay sonrasında yeniden 125/125 geçti. Bu onay eki nedeniyle
`stage1-files.sha256` içindeki yalnız bu inceleme dosyasının hash satırı
yenilendi; özgün onay paketinin hash'i `stage2-approval.json` içinde korunur.
Aşama 2 sonucu ve sonraki testler ayrı `RUN_REPORT.md` kaydındadır.
