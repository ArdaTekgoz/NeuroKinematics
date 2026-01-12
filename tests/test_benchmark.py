import time
import numpy as np
import roboticstoolbox as rtb

# --- DENEY RAPORU İÇİN HIZ TESTİ (Robotics Toolbox) ---
def hiz_testi():
    print("--- NeuroKinSim Benchmark Testi Başlatılıyor (RTB) ---")
    
    # 1. Hazır bir robot modeli yükle (PUMA560 - Endüstriyel Klasik)
    robot = rtb.models.DH.Puma560()
    print(f"Kullanılan Robot: {robot.name}")
    
    # 2. Test parametreleri
    ITERASYON_SAYISI = 100000 
    print(f"Test edilen işlem: İleri Kinematik (Forward Kinematics)")
    print(f"Tekrar sayısı: {ITERASYON_SAYISI}")
    
    # 3. Rastgele eklem açıları üret (6 eklem x 100.000 örnek)
    q_random = np.random.rand(ITERASYON_SAYISI, 6)
    
    # 4. SÜRE BAŞLAT
    start_time = time.time()
    
    # Robotik Toolbox toplu hesaplama yapabilir, ama biz döngü hızını ölçelim
    for i in range(ITERASYON_SAYISI):
        robot.fkine(q_random[i, :])
    
    # 5. SÜRE BİTİR
    end_time = time.time()
    
    gecen_sure = end_time - start_time
    birim_sure_ms = (gecen_sure / ITERASYON_SAYISI) * 1000
    
    print("-" * 30)
    print(f"Toplam Süre: {gecen_sure:.4f} saniye")
    print(f"İşlem Başına Ortalama Süre: {birim_sure_ms:.5f} ms")
    print("-" * 30)
    print("Bu sonucu research_log.md dosyasına kaydet!")

if __name__ == "__main__":
    hiz_testi()