# C1-04 Aşama 2 çalışma kaydı

Kimlik: RUN-20261003-C104-STAGE2

Durum: **IN_PROGRESS / T-C03 PASS / E-C01 üç eşli seed ölçüldü; temiz yeniden üretim ve nihai audit bekleniyor**

Görev ve gereksinim: C1-04 / REQ-C03 / T-C03 / E-C01

Tarih ve sorumlu: 2–3 Ekim 2026 · Codex; proje sahibi Arda Tekgöz

Yazılım hedefi v1.0.0; Aşama 1 belge revizyonu r1 değişmedi.

## Soru ve değişiklik

Kullanıcının açık “Onaylıyorum” yanıtı [approval.json](approval.json) ile Stage1 commit `9b6029bf62feb2caff3b173c600868acc9d57c0c` sonrasına bağlandı. `python scripts/check_c104_stage1.py --check` yeniden 62 girdi/12 dondurulmuş çıktı PASS verdi. Dondurulmuş [config](../config.json), eşikler, veri/split, robot/TCP/FK, model mimarisi ve seedler değiştirilmedi. [Neural kod](../../../src/neurokinematics/neural/c104.py) yalnız train/validation shardlarını okur; test/benchmark model seçim yolunda kapalıdır. Pose-only 7, conditioned 13 özellik; üç 256 SiLU gizli katman, altı mutlak q çıktısı; ortak normalized q MSE, AdamW ve eşli seed sırası. Bağımsız Pinocchio FK validation'da gerçek m/° hatalarını ölçer; ham limit dışı q kırpılmaz.

Gereksinim → değişiklik → test → ham kanıt:

| Gereksinim | Değişiklik | Çalıştırılan test | Kanıt/karar |
|---|---|---|---|
| Şema, maske, sızıntısız özellik | Hash/dtype/split/limit/kanonik quaternion denetimli loader | 12 C1-04 unit/negatif, 24.000 C1-02 verify, gerçek preflight | [JUnit](regression-c104-c102-c103-v2.xml), [preflight](preflight.json); PASS |
| T-C03 küçük gerçek öğrenme ve yanlış eşleme | 64 train/32 ayrı validation local, iki MLP; kaymış etiket FK reddi | 200 epoch; 64/64 yanlış eşleme reddi; loss ve rad q hata oranları | [pilot summary](pilot-summary.json), [preflight](preflight-pilot.json); **T-C03 PASS** |
| E-C01 adil üç seed | Aynı 15.204 train/3.249 validation etiket, sıra, optimizer, 200 epoch ve 3.000 step/model | 2 model ×3 seed, best validation checkpoint | seed summary/epoch JSONL, [audit](audit.json), [eşli özet](E-C01-summary.md); üç seed tamam |
| Gerçek poz ve ham geçerlilik | Bağımsız Pinocchio FK, spot IndependentFK, tüm 3.600 validation satırı | 21.600 per-row sonuç, mod/family/etiket kırılımı, Profile A/B | `seed-*-validation.jsonl`, [özet](E-C01-summary.json); ölçüldü, Profil A 0 |
| Paketlenebilir çıkarım | Metadata/hash kontrollü checkpoint ve 10 sabit validation witness ×6 | Yerel yükleme/çıkarım/FK bit düzeyi tekrar | [witness](fixed-validation-inference.json); PASS, temiz ortam bekliyor |
| Önceki kabul korunumu | C1-02/C1-03 ve Foundations regresyonu | 129 +16+102+159 PASS | JUnit/komut logları; PASS |

## Tekrar üretim

Başlangıç `main`/uzak HEAD Stage1 commit `9b6029bf62feb2caff3b173c600868acc9d57c0c`. Bu raporun uygulama/kapanış commit'i Git geçmişinden okunur. Başlangıçta üç ilgisiz kullanıcı dosyası izlenmiyordu; commit dışı kalır. [Gerçek komutlar](COMMANDS.md), `commands/*/command.json` exact argv/UTC/exit/stdout-stderr base64+SHA, JUnit ve seed JSONL/JSON dosyalarında. İlk pilot/seed terminal stdout'u kayıt aracı eklenmeden koştu; epoch JSONL ve özetler ham sayısal kanıttır. Başarısız ilk hazırlık denemeleri [attempts.json](attempts.json) içinde saklandı.

Gerçek host Windows 11 Pro x64, Ryzen 7 250, Python 3.12.14, Torch 2.10.0+cpu, NumPy 2.5.3, Pinocchio 4.1.0; exact C1-03 hashli overlay, Foundations `pixi.lock` değişmedi. [Ortam](environment.json) kilit ve uygulama SHA'larını içerir. Eğitimde `OMP/OPENBLAS/MKL/NUMEXPR` =1, `torch.set_num_threads(1)`, sıfır worker, CPU float32. C1-02 canonical veri SHA `2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2`; train-only position scaler ve robot limitleri kullanıldı. Veri seed `2026092802`; eğitim seedleri `2026100201`, `2026100202`, `2026100203`. `data/generated/C1-02/v1` shardları ve `data/generated/C1-04/{pilot,v1}` ağırlıkları **LOCAL_ONLY**; uzak arşiv `NOT_CONFIRMED`. Her ağırlığın yol/byte/SHA/erişim bilgisi seed summary dosyasında. Etkin insan emeği **NOT_MEASURED**. Linux/CUDA/fiziksel robot **NOT_RUN**.

## Test ve ham kanıt

T-C03 [pilot](pilot-summary.json): kaymış etiketli 64/64 train kökü bağımsız FK Profile B'de yanlış eşleme olarak reddedildi. Pose-only son/başlangıç train loss oranı `0.04642`, medyan mutlak q hata oranı `0.15273`; conditioned `0.0002571` ve `0.013157`. Dondurulmuş sınırlar ≤0,5 ve ≤0,75; iki model PASS. Validation kaydı tutuldu, pilot train ezberi genelleme olarak yorumlanmadı. İlk Windows RSS probu hata verdi, ilk iki pilot exit1 başarısız hazırlık kaydıdır; sonraki düzeltilmiş pilot 200 epoch/10,438 s ve gözlenen peak RSS 320.487.424 byte ile geçti.

E-C01 tam koşular: her seed/model 200 epoch, 3.000 optimizer step, 15.204 etiketli train ve 3.249 etiketli validation; etkin batch 1024, mikro batch 1024 (birikim gerekmedi). Üç eşli seedin eğitim wall süresi 192,484 / 193,031 / 197,141 s; gözlenen peak süreç RSS 338.243.584 / 338.870.272 / 338.337.792 byte (<4 GiB). Bu Windows CPU ölçümüdür, Linux/CUDA performansı değildir. Her epoch train/validation loss, mod etiket sayısı, rad q hata ve en iyi/son checkpoint kimliği seed JSONL/JSON'da.

| Seed | Model | Best epoch | Validation q loss | Ham limit dışı / 3.600 | Geçerli FK konum medyan (m) | Yönelim medyan (°) | Profil A |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2026100201 | pose-only | 171 | 0,415710 | 162 | 0,5820 | 123,58 | 0/3.600 |
| 2026100201 | conditioned | 199 | 0,170779 | 269 | 0,2061 | 74,65 | 0/3.600 |
| 2026100202 | pose-only | 171 | 0,414716 | 120 | 0,5845 | 121,29 | 0/3.600 |
| 2026100202 | conditioned | 200 | 0,174430 | 184 | 0,2070 | 78,15 | 0/3.600 |
| 2026100203 | pose-only | 165 | 0,414451 | 140 | 0,5706 | 124,12 | 0/3.600 |
| 2026100203 | conditioned | 199 | 0,173756 | 188 | 0,2103 | 76,52 | 0/3.600 |

Tüm FK medyanları yalnız geçerli ham q üzerindendir; payda ve geçersizler [E-C01-summary.json](E-C01-summary.json) içinde. Etiketsiz 351 wide validation satırı envanterde kaldı; onlar için q loss yok, geçerli ham q varsa FK metriği var. C1-06 nihai test ve 10.000 sorguluk benchmark **NOT_RUN**. Kinematik Profile A başarı 0; çarpışma/fiziksel güvenlik kontrolü yapılmadı. 10 sabit validation örneği ×6 checkpoint yerel yükleme/inference/FK tekrarında `max_q_abs_rad=0`, `max_fk_element_abs=0`; temiz ortam kanıtı henüz yok.

Bağımlı testler: C1-04/C1-02/C1-03 birlikte 129 PASS (C1-04 12, C1-02 T-C07 7, C1-03 110), Foundations F0-01/02/03 ayrı 16/102/159 PASS. İlk birleşik pytest toplama denemeleri modül adı çakışması yüzünden exit1; [komutlar](COMMANDS.md) ve hata logları saklandı. Matematik test eşiği değiştirilmedi.

## Sonuç ve yorum

T-C03 **PASS**; E-C01'in üç seedlik adil karşılaştırması ve bağımsız FK/ham geçerlilik ölçümleri tamamlandı. **Kullanılabilir doğrudan IK başarısı yok:** Profile A altı koşunun her birinde 0/3.600. Conditioned q loss ve FK medyanı pose-only'den iyi olsa da fark operasyonel eşik için yeterli değil. [Ambiguity diagnostic](ambiguity-diagnostic.json) pose-only'nin 6.804 etiketli train kökünde aynı pose girdisiyle farklı q hedefi gördüğünü ve medyan iki etiket q L2 farkının 5,815 rad olduğunu ölçer. Bu, pose-only supervised q hedefi için yapısal çelişkidir; conditioned'ın kalan yüksek FK hatasının tek nedeni kesinleşmemiştir. Kullanıcı isteğiyle [sonraki model kararı](NEXT_MODEL_DECISION.md) doğrudan IK kullanımını **NO-GO**, C1-05 E-C03 FK hedefli kontrollü deneyi **sıradaki çözüm** olarak belirler. Checkpointler yalnız izlenebilir araştırma baseline'ıdır.

## Sonraki adım

Yeni checkout ve taze ortamda altı checkpoint yükleme + 10 sabit validation çıkarımı/FK + frozen hash audit'i; sonra evidence manifest ve kapanış kararı. C1-05, C1-04 temiz tekrar ve kabul kaydı bitene kadar **NOT_STARTED**. Düşük başarı saklanır; test sonuçlarına bakılarak C1-04 eşikleri veya seçilmiş checkpointler değiştirilmez. G1/v1.0.0 kapanışı yok.
