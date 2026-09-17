# ADR 002 Kinematik referans ve eğitim katmanı

Durum: Tasarım kararı · Uygulama doğrulaması G0 ve C1-03 içinde

Referans FK/Jacobian için Pinocchio; öğretici bağımsız kontrol için sınırlı seri-zincir FK; eğitimde PyTorch autograd uyumlu FK seçilir. pytorch_kinematics ilk adaydır, küçük Torch FK alternatiftir.

Gerekçe: Referans hesap ve gradyan grafiği farklı sorumluluklardır. İki uygulamanın aynı yanlış URDF'yi kullanabileceği sınırı korunur. Robot geometrisi ayrıca kaynaktan denetlenir.

Seçim desteklenen zinciri karşılamazsa algoritma ve çerçeve testleri korunarak backend değiştirilebilir. Test eşiklerini model sonucuna göre değiştirmek bu ADR'nin alternatifi değildir.
