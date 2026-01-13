import roboticstoolbox as rtb
import numpy as np

class RobotManager:
    def __init__(self):
        # Puma560 Modelini Yükle
        self.robot = rtb.models.DH.Puma560()
        print(f"✅ Robot Yüklendi: {self.robot.name}")

    def get_forward_kinematics(self, joint_angles):
        """Uç nokta (End-Effector) konumunu verir"""
        # Gelen veriyi kesinlikle float numpy dizisine çevir (Hata önleyici)
        q = np.array(joint_angles, dtype=np.float64)
        q_rad = np.deg2rad(q)
        return self.robot.fkine(q_rad)

    def get_all_joint_positions(self, joint_angles):
        """
        Robotun TÜM eklem noktalarının (x,y,z) koordinatlarını verir.
        Bu sayede eklemleri birbirine bağlayan çizgiler (iskelet) çizebiliriz.
        """
        try:
            # Gelen veriyi kesinlikle float numpy dizisine çevir
            q = np.array(joint_angles, dtype=np.float64)
            q_rad = np.deg2rad(q)
            
            # fkine_all: Tabandan uca tüm eklem matrislerini verir (Güvenli yöntem)
            transforms = self.robot.fkine_all(q_rad)
            
            # Her matrisin 't' (translation/konum) kısmını alıyoruz -> [x, y, z]
            points = [T.t for T in transforms]
                
            return np.array(points)
        except Exception as e:
            print(f"Hesaplama Hatası (RobotManager): {e}")
            return None