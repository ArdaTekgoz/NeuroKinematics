# Gelecek araştırma birikimi

Bu işler v0.1–v3.0 kapanışını bloke etmez. Her biri başlamadan ayrı tasarım, veri ve durdurma şartı gerektirir.

| Konu | Giriş şartı | İlk deney | Durdurma veya sınırlama |
|---|---|---|---|
| Generatif çoklu IK | Core/H1 hata analizi dal çeşitliliği ihtiyacı gösterir | Aynı compute altında flow/diffusion ve MLP | Yalnız çeşitlilik artıp geçerlilik düşerse varsayılan yapma |
| MAML ve few-shot | Yeterli farklı robot/görev; ayrık train/val/test robotları | Sıfırdan eğitim ve fine-tune ile aynı veri/bütçede kıyas | Adaptasyon avantajı yoksa Core'u değiştirme |
| GNN | Değişken DoF/topoloji için veri ve ihtiyaç | Yapı bilgili ve per-robot ağ kıyası | Tek robot gösterimini evrensel sayma |
| Gerçek robot ve sim-to-real | Robot erişimi, doğru model/TCP, bağımsız ölçüm sistemi | Kontrollü poz ölçümü ve model hatası | Gürültü enjeksiyonunu gerçek robot testi sayma |
| ROS 2 veya MoveIt servis eklentisi | Sabit SolverRequest/Result ve G2 | Simülasyonda request ve timeout sözleşmesi | Benchmark adaptörünü fiziksel kontrol eklentisi sanma |
| Jetson ve TensorRT | ONNX tamam; ölçülen CPU darboğazı; donanım erişimi | Aynı toleransta FP32/FP16 CPU/GPU kıyası | Hız hedefi yoksa zorunlu donanım alma |
| Collision-aware learning | Post-validation doğru; eğitim geometri dağılımı tanımlı | Aynı başarı bütçesinde collision penalty | Geometri dışı alanda güvenlik iddiası yok |
| Online öğrenme | Sürüm ve regresyon mekanizması, ayrık saha verisi | Offline güncelleme ve eski/yeni görev regresyonu | Kontrol döngüsünde denetimsiz ağırlık değiştirme |
| Dynamics ve RL | Kinematik kapanmış; tork/atalet/temas bilgisi | Açık görevde kontrol ve maliyet kıyası | IK lossundan enerji/aşınma sonucu çıkarma |
| NeuroLocalization | Bağımsız problem, veri ve portföy gerekçesi | Ayrı charter ve baseline | Kinematics'in zorunlu bağımlılığı yapma |
| SaaS ve bulut eğitim | Kullanıcı talebi, maliyet ve veri yetkisi | İsteğe bağlı küçük iş prototipi | Offline çekirdeği buluta bağımlı yapma |

Araştırma başlatma kararı, önceki fazın hangi somut açığını kapatacağı üzerinden verilir. İsim ve marka çeşitliliği tek başına yeni modül gerekçesi değildir.
