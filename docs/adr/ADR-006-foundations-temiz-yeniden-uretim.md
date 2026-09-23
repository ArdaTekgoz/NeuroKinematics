# ADR 006 — Foundations küçük temiz yeniden üretim girdileri

Durum: Kabul edildi
Tarih: 24 Eylül 2026
Yazılım hedefi: v0.1.0 · Belge: r1 · Görev: F0-06

## Bağlam

F0-06, üretim configlerini ve eski kanıtları değiştirmeden küçük veri ve gerçek
DLS benchmark zincirini tekrar üretmeyi gerektirir. Veri fabrikasının public API'si
ayrı config kabul ederken F0-05 query/benchmark API'si üretim yollarını kullanıyordu.

## Karar

Query üretim/doğrulama ve benchmark public API'lerine birlikte verilmesi gereken
`config_path`, `dataset_manifest`, `data_config` parametreleri eklendi. Varsayılan
üretim yolu korunur. Ayrı benchmark config'i yalnız örnek sayısı, tekrar sayısı
ve ısınma sayısını azaltabilir. Seedler, solver, toleranslar, schema ve deadline
sözleşmeleri aynıdır; dondurulmuş F0-04/F0-05 dosyaları yeniden yazılmaz.

`scripts/reproduce_f06.py` gerçek veri fabrikası, query üretici, DLS runner ve
bağımsız doğrulayıcıyı çağırır. Mock, monkeypatch veya fixture ile kabul yapılmaz.
Küçük dataset 256/64/64, query 256/64/64 ve benchmark 2 deadline × 1 geçiştir.
Bu smoke ölçeği üretimdeki 10000/1000/1000 kabul deneylerinin yerine geçmez.

Dataset tekillik eşiği küçük main train bölümünden hesaplanır. Query tekillik
eşiği mevcut F0-05 üretim train referansından dondurulmuş olarak devralınır.
Query/test sonuçlarından eşik seçilmez; normalizasyon yalnız main train'dendir.

## Kanıt ve sürümleme

Önce F0-05 kapanış commit'i temiz worktree'de sınanır. Sonra F0-06 kaynak
farkları `overlay.json` ile hashlenerek yeni bir temiz worktree'ye uygulanır.
Önceki ortam, pytest cache veya çıktı dizinleri kopyalanmaz. Global Pixi indirme
önbelleği kullanılabilir ve kayda yazılır. Yeni bağımlılık ve lock değişikliği yoktur.

Eski checksum kayıtlarının işaret ettiği paylaşılan dosyalar sonraki görevlerde
değişmişse eski ham Git blob'u ayrı doğrulanır; güncel dosyanın eski hash ile
eşleştiği söylenmez. F0-06 source overlay'i de başlangıç blob'u ve yeni hash ile
ayrı kaydedilir. Immutable robot/config/schema dosyalarına bu istisna uygulanmaz.

## Sonuçlar

Yeni IK veya Core işi yoktur. İki bağımsız koşuda veri/query/shard hashleri
karşılaştırılır. Benchmark süreleri, iterasyonlar ve deadline'a bağlı durum
frekansları Windows zamanlamasıyla değişebilir. Deterministik kimlikler ve
her koşunun gerçek schema/limit/manifest bağları karşılaştırılır. Manifestteki
üretim yolu farklı olduğundan ham manifest hashleri kendi koşusuna bağlanır.

Factory manifestindeki eski varsayılan üretim komutu smoke config'ini içermez;
F0-06 için bağlayıcı komut `COMMANDS.md` ve `commands.json` içindeki açık
`reproduce_f06.py` çağrısıdır. Eski manifest biçimi sessizce yeniden tanımlanmaz.
