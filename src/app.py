import sys
import time
import os
import ctypes
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from gui.splash_screen import NeuroSplash
from gui.main_window import IndustrialRobotGUI


def main():
    """NeuroKinematics Ana Giriş Noktası"""
    
    # --- WINDOWS GÖREV ÇUBUĞU İKONU DÜZELTMESİ ---
    # Windows'a "Bu python değil, NeuroKinematics uygulamasıdır" diyoruz.
    if sys.platform == "win32":
        myappid = 'ardatekgöz.neurokinematics.gui.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    
    # 1. Uygulamayı Başlat
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # --- UYGULAMA GENELİ İKONU AYARLA ---
    # Burası hem pencere sol üst köşesini hem de taskbar'ı değiştirir
    # Logo path'i: src/ klasöründen bir üst dizine çıkıp assets/logo.png'ye git
    logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")
    if os.path.exists(logo_path):
        app.setWindowIcon(QIcon(logo_path))
    
    # 2. Splash Ekranını Göster
    splash = NeuroSplash()
    splash.show()
    
    # 3. Yükleme Simülasyonu
    for i in range(1, 101):
        time.sleep(0.015)
        splash.showMessage(
            f"Loading Modules... {i}%",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
            Qt.GlobalColor.white
        )
        app.processEvents()
    
    # 4. Ana Pencereyi Hazırla
    window = IndustrialRobotGUI()
    
    # Pencereye de ikon set edelim (Garanti olsun)
    if os.path.exists(logo_path):
        window.setWindowIcon(QIcon(logo_path))
    
    # 5. Splash'i Kapat ve Ana Pencereyi Aç
    splash.finish(window)
    window.show()
    
    # 6. Döngüyü Başlat
    sys.exit(app.exec())


if __name__ == "__main__":
    main()