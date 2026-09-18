# ADR 004 Foundations platformu ve kapasite varsayımları

Durum: Kabul edildi

Karar tarihi: 18 Eylül 2026

Etkilenen hedef: Foundations v0.1.0

## Bağlam

Plan araştırma ortamı olarak Linux'u öneriyordu; ancak F0-00 başlangıç hostunda WSL, ayrı Linux, Docker ve Conda kurulu değildir. Host Windows 11 Pro x64 10.0.26200'dır. Pinocchio 4.1.0'ın pip paketi Windows wheel'i sunmaz; buna karşılık upstream proje `win-64` üzerinde Pixi CI'ı çalıştırır ve conda-forge paketi sağlar.

F0-00'ın amacı hayalî bir hedef ortamı belgelemek değil, kilitli ortamı gerçekten kurup küçük kontrolü çalıştırmaktır.

## Karar

- Foundations için kanonik geliştirme ve test hostu native Windows 11 x64'tır.
- Ortam Pixi 0.81.0 ve conda-forge ile yönetilir; WSL veya sistem Python'u proje ortamı olarak kullanılmaz.
- CPython 3.12 serisi, Pinocchio 4.1.0, NumPy ve pytest doğrudan bağımlılıklardır. Exact transitif çözüm `pixi.lock` ile sürümlenir.
- Lock hedefleri `win-64` ve `linux-64` olur. Linux çözümünün bulunması Linux'ta test çalıştırıldığı anlamına gelmez.
- İlk doğrulama CPU üzerinde ve float64 yapılır. GPU varlığı kabul şartı veya performans iddiası değildir.
- Windows hostunda doğrulanmayan ROS/KDL/TRAC-IK ihtiyacı doğarsa ilgili görev başlamadan Linux/WSL kararı yeni ADR veya bu ADR'nin yeni revizyonuyla alınır.
- Studio'nun dağıtım platformu kararı bu Foundations kararından ayrı kalır.

## Gerekçe

Bu seçim mevcut hostta hemen çalıştırılabilir, ek sanallaştırma/reboot gerektirmez ve referans backend'in upstream Windows test yoluyla uyumludur. Pixi, Python ile native bağımlılıkları aynı kilitte çözer; bu nedenle Windows üzerinde desteklenmeyen `pip install pin` yoluna güvenilmez.

## Sonuçlar

- Native Windows F0-00, F0-02 ve sonraki CPU kontrollerinin birincil kanıt platformudur.
- Yol ayırıcıları, dosya kodlaması ve multiprocessing davranışı Windows'ta test edilmelidir.
- Linux yalnız lock çözüm hedefidir; çalıştırılana kadar destek durumu `NOT_RUN` kalır.
- WSL'nin kurulu olmaması F0-00 için engel değildir.
- Donanım bağımlı benchmark sonuçları OS/CPU/RAM/thread bilgisiyle adlandırılır.
- Haftalık 8–12 saat kapasite planlama varsayımıdır; teknik test sonucu değildir.
- Yeni donanım, bulut harcaması ve fiziksel robot alımı bu kararla yetkilendirilmez.

## Yeniden değerlendirme tetikleri

- Pinocchio `win-64` paketi veya kilitli transitif bağımlılıkları kurulamazsa;
- zorunlu ROS tabanlı backend native Windows'ta desteklenmiyorsa;
- Windows/Linux sayısal sonuçları kabul eşiğini etkileyen biçimde ayrışırsa;
- temiz ortam tekrarı native Windows'ta başarısız olursa.

Bu tetiklerden biri oluşursa başarısız sonuç gizlenmez; doğrulanmış WSL2 Ubuntu 24.04 LTS veya ayrı Linux ortamına geçiş kaydedilir.
