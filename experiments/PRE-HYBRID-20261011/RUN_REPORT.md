# Deney veya uygulama kaydı

Kimlik: PRE-HYBRID-20261011  
Durum: PASS / LOCAL_DELIVERY_COMPLETE; bağımsız aygıt NOT_CONFIRMED  
Görev ve gereksinim: Hybrid öncesi checkpoint, doğru durum, güncel belgeler, devralınabilirlik  
Tarih ve sorumlu: 11 Ekim 2026 · Arda Tekgöz / Codex

## Soru ve değişiklik

Kullanıcının beş teslim sorusundaki eksikler gideriliyor. Güncel giriş belgeleri,
tarihsel SHA kopyaları, tek devir rehberi ve Git + büyük artifact arşivi hazırlanır.
Algoritma/eşik değiştirilmez; neural hedef ve çalıştırılmayan Hybrid testleri açık kalır.
ADR-028 bu kanıt koruma ve teslim ayrımını kaydeder.

## Tekrar üretim

Başlangıç HEAD: 0f518d24dadd83b4e7856f6cd2eac3bee17d2da2, branch codex/c1-06r.
Kirli çalışma dizisindeki C1-06S/r3/doküman işleri kapsama alınır; önceden mevcut
kullanıcı STATUS/TRACE değişiklikleri ve ilgisiz kaynak belgeleri commit dışı korunur.
Komut/ortam/sonuç kayıtları bu klasördeki JSON ve stdout/stderr dosyalarında tutulur.
Model, robot, veri, protokol kimlikleri orijinal devir manifestlerinden doğrulanır.

## Test ve ham kanıt

Beklenen: tarihsel hashler aynı; aktif belge linkleri/DAG geçerli; bundle clone tam;
artifact bütün üye SHA ve boş hedef restore aynı; seçilen ikinci kopya doğrulanmış.
Yeni eğitim/benchmark/fresh environment/collision/ONNX/Hybrid testleri NOT_RUN.
Önceki 315 test ve 12.000 geometrik final sonuçları yeni ölçüm olarak sayılmaz.

## Sonuç ve yorum

Git checkpoint, archive restore, temiz kaynak doğrulaması ve seçilen ikinci kopya tamamlandı. Aynı C: diskindeki
ikinci kopya bağımsız aygıt sayılmaz; independent device NOT_CONFIRMED kalır.

## Sonraki adım

Checkpoint doğrulandıktan sonra docs/tasks/H2-01.md uygulanır. Neural eğitim
H2-N1/N2, katkı H2-03; G2/G-N ayrıdır. Hiçbir negatif sonuç PASS'e çevrilmez.


## Gerçekleşen teslim ve doğrulama

- İlk kapsam checkpoint'i `2ccf85c`; restore düzeltmesi `f4eff1f`; frozen kaynak
  satır sonu koruması `6591c8a`. Son kanıt commit'i Git geçmişinde bu raporu izler.
- `historical-snapshots.json`: 8 giriş/durum/doğrulayıcı dosyası aynı baytlı tarihsel kopya.
  C1-07/C1-06S/r3 özgün manifestleri değişmedi. ADR-028 yeni devir kapsamını açıklar.
- `integrity.json`: 450 hash/politika kontrolü PASS. `clone-integrity-r3.json` temiz
  bundle klonunda aynı 450 kontrol PASS. Model kullanım DOCX kaynak/çıktı SHA ayrıca
  son doğrulayıcıya eklendi; nihai doğrulama kaydında 452 kontrol beklenir.
- `clone-plan-r3.json`: 1050 kontrol PASS, 39 görev, 369 aktif bağlantı; başlangıçta
  yalnız H2-01 hazır. H2 uygulaması ve eğitim NOT_STARTED.
- `archive-create.json`/`archive-restore.json`: 612 dosya, 6.128.837.042 ham bayt,
  1.055.999.341 ZIP baytı; tüm member SHA/boyutları ve boş hedef restore PASS.
  Arşiv oluşturulurken HEAD önceki tarihsel commit'ti ve çalışma ağacı kirliydi;
  source_head bu yaratım anını gösterir. Teslim kod kimliği bundle/verification'dadır.
- `clone-install-r3.json`: 482 eksik artifact eklendi, 130 aynı SHA'lı mevcut dosya
  korundu; overwrite 0. Git bazı eski pilot ham kayıtları zaten içerir.
- `clone-geometric-tests-r3.xml`: 38 test PASS, skip/fail/error 0.
  `restored-replay-r3.json`: 12.000 sorgu, EXACT candidate/status ve FK metrics.
  Mevcut kilitli Python overlay kullanıldı; fresh environment NOT_CREATED.
- `secondary-copy.json`: kullanıcının Documents/NK_Arşiv hedefindeki ilk tam
  kopyanın 220 dosyası kaynak SHA'larıyla aynı. Nihai bundle ve copy makbuzu
  arşiv kökündeki `DELIVERY.json` ile ayrıca doğrulanır; bu raporun kendi commit'i
  rapor içine kendi hash'ini yazamaz. Son canonical repository.bundle bu commit'i içerir.
- git diff --cached --check PASS; C1-06S/plan/model/belge/source PDF'ler ve bu
  çalışma dizisindeki küçük yayın çıktıları checkpoint'te. Yayın/push/tag yok.
- `scope.json`: önceki kullanıcı STATUS/TRACE audit değişiklikleri index dışında
  bırakıldı ve çalışma ağacında korunuyor; ilgili kullanıcı belgeleri/AUDIT dizini
  arşiv `user-work/` içinde aynı baytlarla saklanıyor. Word geçici kilit dosyası alınmadı.

## Denemede bulunan taşınabilirlik kusurları

İlk clone restore mevcut pilot JSONL nedeniyle reddedildi: no-overwrite kuralı
doğru çalıştı. Boş artifact-store restore + SHA denetimli install eklendi.
İkinci install ve temiz hash denetimi Git autocrlf nedeniyle sırasıyla pilot
JSONL ve 14 eski kaynak dosyasında farklı SHA buldu. `.gitattributes` raw ve
frozen girdilerin exact baytlarını koruyacak şekilde sabitlendi. Son temiz
clone install/hash/plan/test/replay PASS; eski başarısız loglar silinmedi.
Hazırlık sırasında Python path literal hatası ve ignored data'yı stage etmeye
çalışan yardımcı komut hatası düzeltildi; ignored veri Git'e zorla eklenmedi.

## Kullanıcının beş sorusunun kapanışı

1. Yerel sağlam checkpoint: kod+Git bundle ve büyük artifact restore doğrulandı.
2. G0/G1 araştırma ve geometrik kabul tamam; direct neural NOT_MET, H1 NOT_MEASURED.
   Bu açık hedefler H2-N1/N2/03 ve ayrı G-N kapısına bağlı; bütün metrikler PASS denmedi.
3. README, AGENTS, geri dönüş rehberi, durum/trace ve girişler güncel roadmap ile tutarlı.
4. Devir/checkpoint/restore işleri tamam; kalan koruma farklı aygıt veya uzak backup.
5. START_HERE okuma sırası, iki motorun rolü, kurulum/restore, ilk H2-01 görevi,
   sınırlar ve kabul kuralları başka kişi/model için açık. Git clone tek başına yeterli değil.

Bağımsız aygıt veya uzaktaki kopya NOT_CONFIRMED; iki kopya aynı C: diskinde.
Neural eğitim, H1 ölçümü, yeni benchmark, fresh ortam, collision, ONNX ve fiziksel
güvenlik NOT_RUN; bu belge/teslim kapanışı bunların yerine geçmez.
