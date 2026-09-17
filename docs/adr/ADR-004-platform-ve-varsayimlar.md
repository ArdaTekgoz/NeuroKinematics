# ADR 004 Robot platform ve kapasite varsayımları

Durum: Planlama varsayımı; uygulama başlangıcında gözden geçirilecek

İlk robot KUKA KR6 R900 sixx, ikinci aday UR5; çalışma kapasitesi haftada 8–12 saat. Araştırma ortamı Linux; Studio ilk dağıtım hedefi Windows 11 x64 CPU. Kesin robot varlıkları, OS sürümü ve bağımlılık kilidi henüz edinilip test edilmedi.

F0-00/F0-01 ortam ve model kararını, S3-02 Windows prototipini kapatır. Kullanıcının kapasitesi veya makinesi farklıysa iş saatleri korunup takvim yeniden hesaplanır. Yeni donanım, bulut harcaması ve fiziksel robot alımı yetkilendirilmiş/planlanmış harcama değildir.

Kendi kodunun açık kaynak lisansı ve kamuya ilk yayın tarihi release öncesi ayrıca kararlaştırılır. Bu plan repository lisansını veya erişimini değiştirmez.
