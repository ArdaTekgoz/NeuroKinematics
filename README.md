# NeuroKinematics

Derin Öğrenme Tabanlı Modüler Robotik Ters Kinematik Çözücü ve Yörünge Planlama Platformu.

## Proje Hakkında
NeuroKinematics, klasik ters kinematik yöntemlerinin fiziksel güvenilirliğini korurken, öğrenme tabanlı yöntemlerin (Deep Learning) hızlı çıkarım ve genelleyebilme potansiyelinden yararlanabilecek "Fizik Farkındalıklı" (Physics-Aware) bir IK (Inverse Kinematics) mimarisi geliştirmeyi amaçlar.

## Klasör Yapısı
- `neurokinematics/core/`: İleri kinematik, Jacobian ve robot modelleme (Pinocchio tabanlı)
- `neurokinematics/data/`: Sentetik veri üretimi ve veri işleme
- `neurokinematics/model/`: Res-MLP ve yapay zeka modelleri
- `neurokinematics/sim/`: 3D Görselleştirme, GUI ve Dijital İkiz (PyVista, PyQt)
- `neurokinematics/utils/`: Yardımcı araçlar, loglama vb.
- `tests/`: Birim testleri
- `notebooks/`: Araştırma ve prototipleme amaçlı Jupyter not defterleri
- `experiments/`: Eğitim metrikleri ve model çıktıları

## Kurulum
```bash
pip install -r requirements.txt
```
