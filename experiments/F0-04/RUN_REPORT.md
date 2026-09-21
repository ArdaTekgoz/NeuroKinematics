# F0-04 deterministik veri fabrikası çalışma kaydı

Kimlik: RUN-20260921-001
Durum: PASS · KABUL · TAMAMLANDI
Görev ve gereksinim: F0-04 · REQ-F04 · T-F05 / T-F06 / T-F07
Tarih ve sorumlu: 21 Eylül 2026 · Codex; proje sahibi Arda Tekgöz
Yazılım hedefi: v0.1.0 · Belge revizyonu: r2

## Soru ve değişiklik

Sonuçlar görülmeden dondurulan PCG64 seedleri ve coverage gridleri ile aynı
typed-array dataset iki temiz klasörde byte düzeyinde yeniden üretilebiliyor mu;
grup/split sızıntısı sıfır mı; Pinocchio etiketleri bağımsız FK ve doğrulanmış
Jacobian/SVD metrikleriyle uyumlu mu? `src/neurokinematics/data/` altında LHS,
split, canonical quaternion/hash, sabit ZIP metadata'lı NPZ shard, doğrulama,
hard-subset ve ampirik coverage katmanı eklendi. Kinematik modülleri veri modülünü
import etmez. Eğitim, DLS, Torch, ONNX, GUI ve F0-05 başlatılmadı.

Config kod çalıştırılmadan önce donduruldu. Ana seed `20260920`; split
`20260921`, uniform `20260922`, boundary `20260923`, singular candidate
`20260924`. Config SHA-256
`baeee1b574561f218eef5b4fa0bee662478ca07527c5fe832499cafbaf5760ae`,
schema SHA-256
`392ddf0d057f81b59d5851d514ef384513702abc2a091d8e79b88b58ffafa36e`.
Sonuçlara göre seed, eşik, örnek sayısı veya grid değiştirilmedi.

## Tekrar üretim

| Alan | Değer |
|---|---|
| Başlangıç HEAD | `85897aad33a5c5c91442859bc6ea0be2809ee1ec`; `main == origin/main` |
| Uygulama commit'i | `16010d518c24400f6c6d43a2459456dd822f34a8` |
| Ortam | Windows 11 x64; Python 3.12.14; NumPy 2.5.3; Pinocchio 4.1.0; Linux NOT_RUN |
| Lock | Değişmedi; `56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a` |
| Veri | main 10000 LHS; boundary 1000; singularity 1000; uniform yalnız coverage için 10000 |
| Shard | main 10 × 1000; boundary 1 × 1000; singularity 1 × 1000; full shardlar Git dışında |
| Dataset content | `5cb4e64580ecaf99afd11b3c8b98e06ed00c712e83bf2d9ee4d8c3acd58173fe` |
| Üretim | `pixi run --locked python scripts/run_f04_acceptance.py` |
| Ayrı tekrar | `--generated-root data/generated/F0-04-reproduction --output temp/f04-evidence` kullanılabilir |

Main shard content hashleri sırasıyla:

`d0bb96fef1991c668447ed69ea6373c6dd994987be51c67ca84e4a370ffa94c1`,
`251c55fa5ad43931b40cd0a72857584c0c3ec5abea7e36b56120626a71338cd6`,
`cdf2866021218ff7b1f3c4d9d1f96fe45862a04c58ee266c67e31373cb66ae29`,
`7eaca4f99e1b3d75546e6803ddaad7d0ff9d52690d9dea89165f894c3b2fa196`,
`273a710b083d4f76ae303bb7e92e0281a49f5bcd5935b7f00299234b41908ad0`,
`371b9f21f88d36d1920c0e83bb6ddf4c416db0c11031d77b4ac768f73fa452da`,
`14af9e01e0610dad7142f282208e5d98677b906bf4125da1dfd3ed3690465ca2`,
`6c8de89c460c2b9044097da9331ff201f830ba86d835e1ff0a78afc7337b1c38`,
`1cdce175b7ffa9b2bb98555a612aae7fa39e4228b195f3c0fa7513d160854085`,
`ebc2fff5b7f5fc33318644d570ee287d108e13be403655d50ac3c632890b702e`.
Boundary: `57b0163569d5d922c249eafd3b1f11005477e37390be026eac4bc4ba4493eb3b`;
singularity: `a085434347b7571b18241f96761eb33a7fa4f170c4239e468d40a59f4ccc7039`.
İki bağımsız üretimde tüm 12 file ve content hash ile dataset hash birebir eşti.

## Test ve ham kanıt

| Kontrol | Ölçülen sonuç | Kanıt |
|---|---|---|
| F0-00/01/02/03 regresyon | 6/6, 16/16, 102/102, 159/159 PASS | ilgili JUnitler |
| F0-04 unit / T-F05 / T-F06 / T-F07 | 10/10, 3/3, 2/2, 7/7 PASS | F0-04 JUnitler |
| Mutasyon | 17/17 yakalandı | `mutation-results.json`, JUnit properties |
| Determinizm | 2 temiz üretim; 12/12 file/content ve dataset hash eş | `determinism-summary.json` |
| Split | main 7000/1500/1500; hard subsetlerin her biri 700/150/150 | `split-audit.json` |
| Grup kesişimleri | train/validation, train/test, validation/test: 0/0/0 | `split-audit.json` |
| Duplicate/örtüşme | çapraz split q 0/0/0; main/boundary/singular 0/0/0 | `duplicate-audit.json` |
| Train-only normalizasyon | 7000 train kayıt; validation/test değişimi sonucu etkilemiyor | `normalization.json`, unit JUnit |
| FK doğruluğu | max position `6.69794233840692e-16 m`; rotation Frobenius `9.46264227718545e-16` | `fk-validation-summary.json` |
| Boundary | 1000; strict `<0.02`; gözlenen max `0.0199999437235451`; joint/taraf dengesi kayıtlı | `hard-subsets-summary.json` |
| Singularity | train yüzde 5 eşiği `0.00727741160967353`; 20373 candidate; oran `0.0490845727187945` | `hard-subsets-summary.json` |
| Kanıt bütünlüğü | 39 dosya doğrulandı | `SHA256SUMS`, `evidence-verification.json` |

Birincil coverage (`joint / position / orientation / combined pose` occupancy):
LHS `9950 / 7626 / 7844 / 9998`; uniform `9955 / 7630 / 7815 / 10000`.
LHS combined pose prefix eğrisi `1000, 2500, 5000, 7499, 9998`; uniform
`1000, 2500, 5000, 7500, 10000`. İnce profilde combined pose iki havuz için
`10000/10000`; birincil `9998/10000`; kaba `9987/9990`. Position occupancy
ince/birincil/kaba olarak LHS `9532/7626/3254`, uniform `9530/7630/3256`.
Bu değerler yalnız önceden tanımlı ampirik örnek havuzlarının occupancy'sidir;
erişilebilir fiziksel çalışma uzayının yüzdesi değildir.

| Gereksinim | Kod | Test | Kanıt |
|---|---|---|---|
| Deterministik LHS/shard/hash | `factory.py` | unit, T-F05 | manifest, determinism summary |
| Grup split ve train-only normalizasyon | `factory.py` | T-F06, mutation | split/duplicate/normalization JSON |
| Referans FK ve Jacobian/SVD | F0-02/03 servisleri + data factory | T-F07 | FK ve hard-subset özetleri |
| Boundary/singularity | frozen config + `factory.py` | T-F07, mutation | hard-subsets summary |
| Coverage/sensitivity | frozen config + `coverage()` | T-F07 | coverage JSONları |
| Regresyon ve bütünlük | `run_f04_acceptance.py` | tam runner | commands/JUnit/SHA256SUMS |

## Sonuç ve yorum

**F0-04 PASS / KABUL / TAMAMLANDI.** T-F05, T-F06 ve T-F07 önceden
dondurulmuş kararlarla geçti. NaN/nonfinite, joint-limit ihlali, yanlış frame/sıra,
bozuk quaternion veya FK eşik aşımı yok. `condition=Inf` yalnız exact
`sigma_min=0` sözleşmesine bağlıdır. Shard bozulması/eksilmesi ve config/schema
uyuşmazlığı reddedilir.

Veri yalnız immutable model içindeki kinematik doğruluğa sahiptir; fiziksel robot
doğruluğu, kalibrasyon, collision-free davranış veya safety certification değildir.
Coverage ampirik örnek havuzu ölçüsüdür. Linux çalıştırılmadı. Büyük dataset
Git'e eklenmedi; iki yerel üretim klasörü `.gitignore` kapsamındadır.

## Sonraki adım

F0-04 kapanmıştır. F0-05 için veri manifesti ve train-only normalizasyon girdileri
hazırdır; bu çalışma F0-05'i başlatmamıştır. Foundations G0 henüz kapanmamıştır.
