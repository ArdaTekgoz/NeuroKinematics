# NeuroKinSim - Araştırma Günlüğü

## Tarih: 13.01.2026
**Deney:** Sayısal Yöntem Performans Testi (Baseline)
**Amaç:** Klasik yöntemlerin (Robotics Toolbox / C++ Backend) işlem hızını belirlemek.

### Sonuçlar
* **Donanım:** AMD Ryzen 7 250 w/ Radeon 780M, Nvidia RTX5060, 24GB RAM, 2TB SSD
* **Yöntem:** Robotics Toolbox (Puma 560 Modeli)
* **İşlem:** İleri Kinematik (FK)
* **Tekrar Sayısı:** 100.000
* **Ortalama Süre:** 0.09243 ms (İşlem başına)

**Notlar:**
Bu değer, geliştireceğimiz Nöral Ağ (Res-MLP) için referans hız kabul edilecektir. Yapay zeka modelinin bu süreye yakınsaması (inference time) hedeflenmektedir.