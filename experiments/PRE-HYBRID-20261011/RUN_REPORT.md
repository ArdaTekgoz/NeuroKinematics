# Deney veya uygulama kaydı

Kimlik: PRE-HYBRID-20261011  
Durum: IN_PROGRESS  
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

Gerçek test ve arşiv sonuçları tamamlanınca bu kayda eklenecek. Aynı C: diskindeki
ikinci kopya bağımsız aygıt sayılmaz; independent device NOT_CONFIRMED kalır.

## Sonraki adım

Checkpoint doğrulandıktan sonra docs/tasks/H2-01.md uygulanır. Neural eğitim
H2-N1/N2, katkı H2-03; G2/G-N ayrıdır. Hiçbir negatif sonuç PASS'e çevrilmez.
