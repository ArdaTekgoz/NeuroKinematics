import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSlider, QLabel, QFrame, QMenuBar, QMenu, QGroupBox, QLineEdit,
    QRadioButton, QDial, QFileDialog, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut, QAction
import pyvista as pv
from pyvistaqt import QtInteractor

# --- GÜVENLİ ROBOT BAĞLANTISI ---
try:
    from src.core.robot_manager import RobotManager
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False

class IndustrialRobotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 1. Robot Yöneticisini Başlat
        self.robot_manager = None
        self.current_joints = [0.0] * 6
        self.robot_mesh = None  # Robot stick figure mesh
        self.plotter = None     # Plotter referansı
        
        if ROBOT_AVAILABLE:
            try:
                self.robot_manager = RobotManager()
                print("✅ SİSTEM: Robot beyni devreye girdi.")
            except Exception as e:
                print(f"❌ HATA: Robot başlatılamadı: {e}")

        # 2. Pencere Ayarları
        self.setWindowTitle("NeuroKinematics - NextGen Industrial Robot Control System")
        self.resize(1400, 900)
        
        # 3. Dil ve Ayarlar
        self.current_language = "en"
        self.current_euler_convention = "Z-Y-X"
        self.tcp_trace_enabled = False
        self.dark_mode = True
        
        self.panel_visibility = {
            "robot_control": True, "objects": False, "simulation": False, "abb_program": False
        }
        
        # --- ÇEVİRİ SÖZLÜĞÜ ---
        self.translations = {
            "en": {
                "window_title": "NeuroKinematics", "robot_controls": "Robot Controls", "joint": "Joint",
                "file": "File", "view": "View", "options": "Options", "help": "Help", "language": "Language",
                "exit": "Exit", "light_mode": "Light Mode", "dark_mode": "Dark Mode"
            },
            "tr": {
                "window_title": "NeuroKinematics", "robot_controls": "Robot Kontrolleri", "joint": "Eklem",
                "file": "Dosya", "view": "Görünüm", "options": "Seçenekler", "help": "Yardım", "language": "Dil",
                "exit": "Çıkış", "light_mode": "Aydınlık Mod", "dark_mode": "Karanlık Mod"
            }
        }
        self.camera_names = {"en": ["Isometric", "Top"], "tr": ["İzometrik", "Üstten"]}

        # 4. Arayüzü Kur
        self.setup_ui()
        
        # 5. Görselleştirmeyi Başlat
        if self.robot_manager and self.plotter is not None:
            self.init_robot_visualization()
            self.update_robot_state()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.left_panel = self.create_left_panel()
        main_layout.addWidget(self.left_panel)
        
        self.center_panel = self.create_center_panel()
        main_layout.addWidget(self.center_panel, 1)
        
        self.right_panel = self.create_right_panel()
        main_layout.addWidget(self.right_panel)
        
        self.create_menu_bar()
        self.apply_stylesheet()

    def create_left_panel(self):
        panel = QFrame()
        panel.setObjectName("leftPanel")
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)
        
        self.left_title = QLabel("Robot Controls")
        self.left_title.setObjectName("panelTitle")
        self.left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.left_title)
        
        self.joint_labels = []
        for i in range(6):
            row = QHBoxLayout()
            name_lbl = QLabel(f"Joint {i+1}")
            val_lbl = QLabel("0°")
            val_lbl.setObjectName("valueLabel")
            self.joint_labels.append(val_lbl)
            
            row.addWidget(name_lbl)
            row.addStretch()
            row.addWidget(val_lbl)
            layout.addLayout(row)
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-180, 180)
            slider.setValue(0)
            slider.valueChanged.connect(lambda val, idx=i: self.update_joint_value(idx, val))
            layout.addWidget(slider)
            
        layout.addStretch()
        return panel

    def create_center_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        try:
            # FIX: QtInteractor'ı önce oluşturup sonra layout'a ekliyoruz
            self.plotter = QtInteractor() 
            layout.addWidget(self.plotter.interactor)
            
            self.plotter.set_background("#101018")
            self.plotter.add_axes()
            # FIX: add_grid yerine show_grid kullanıyoruz
            self.plotter.show_grid(color="#333333")
            self.plotter.view_isometric()
            
        except Exception as e:
            print(f"⚠️ PyVista Hatası: {e}")
            self.plotter = None
            
        return container

    def create_right_panel(self):
        panel = QFrame()
        panel.setObjectName("rightPanel")
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)
        
        self.right_title = QLabel("Kinematics Data")
        self.right_title.setObjectName("panelTitle")
        self.right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.right_title)
        
        layout.addWidget(QLabel("End-Effector Position (X, Y, Z):"))
        self.pos_edit = QLineEdit()
        self.pos_edit.setReadOnly(True)
        layout.addWidget(self.pos_edit)
        
        layout.addStretch()
        return panel

    def create_menu_bar(self):
        menubar = self.menuBar()
        self.file_menu = menubar.addMenu("File")
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        self.file_menu.addAction(exit_act)
        
        self.view_menu = menubar.addMenu("View")
        self.theme_act = QAction("Light Mode", self)
        self.theme_act.triggered.connect(self.toggle_theme)
        self.view_menu.addAction(self.theme_act)

    # --- MANTIK VE GÖRSELLEŞTİRME ---

    def update_joint_value(self, index, value):
        self.joint_labels[index].setText(f"{value}°")
        self.current_joints[index] = float(value)
        self.update_robot_state()

    def init_robot_visualization(self):
        """Robot iskeletini ilk kez oluşturur"""
        if not self.robot_manager or not self.plotter:
            return
        
        try:
            # FIX: NoneType kontrolü eklendi
            points = self.robot_manager.get_all_joint_positions(self.current_joints)
            if points is None: 
                print("Robot verisi alınamadı!")
                return
            
            # PolyData mesh oluştur
            self.robot_mesh = pv.PolyData()
            self.robot_mesh.points = points
            
            # Noktaları çizgiyle bağla
            n_points = len(points)
            lines = [n_points] + list(range(n_points))
            self.robot_mesh.lines = lines
            
            # Sahneye ekle
            self.plotter.add_mesh(
                self.robot_mesh, 
                color="#14ffec", 
                line_width=5, 
                render_lines_as_tubes=True,
                point_size=15,
                render_points_as_spheres=True,
                name="robot_skeleton"
            )
        except Exception as e:
            print(f"Görselleştirme başlatma hatası: {e}")

    def update_robot_state(self):
        if not self.robot_manager:
            return
            
        try:
            # 1. Yeni noktaları al
            new_points = self.robot_manager.get_all_joint_positions(self.current_joints)
            
            # FIX: NoneType kontrolü
            if new_points is None:
                return

            # 2. İskeleti güncelle
            if hasattr(self, 'robot_mesh') and self.robot_mesh is not None:
                self.robot_mesh.points = new_points
                if self.plotter:
                    self.plotter.render()
            
            # 3. Metin kutularını güncelle
            if len(new_points) > 0:
                end_pos = new_points[-1]
                if hasattr(self, 'pos_edit'):
                    self.pos_edit.setText(f"{end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f}")
                
        except Exception as e:
            print(f"Güncelleme hatası: {e}")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_stylesheet()

    def apply_stylesheet(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow { background-color: #101018; color: white; }
                QMenuBar { background-color: #2d2d2d; color: #e0e0e0; }
                QMenu { background-color: #2d2d2d; color: #e0e0e0; }
                QFrame#leftPanel, QFrame#rightPanel { background-color: #252526; border-right: 1px solid #333; }
                QLabel#panelTitle { font-size: 16px; font-weight: bold; color: #14ffec; border: 1px solid #14ffec; border-radius: 5px; padding: 10px; }
                QLabel { color: #e0e0e0; }
                QLineEdit { background-color: #333; color: #14ffec; }
                QSlider::handle:horizontal { background-color: #14ffec; }
            """)
            if self.plotter: self.plotter.set_background("#101018")
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #f0f0f0; color: black; }
                QMenuBar { background-color: #fff; color: black; }
                QFrame#leftPanel, QFrame#rightPanel { background-color: #fff; border-right: 1px solid #ccc; }
                QLabel#panelTitle { font-size: 16px; font-weight: bold; color: #007acc; border: 1px solid #007acc; border-radius: 5px; padding: 10px; }
                QLabel { color: black; }
                QLineEdit { background-color: #fff; color: black; }
                QSlider::handle:horizontal { background-color: #007acc; }
            """)
            if self.plotter: self.plotter.set_background("#f0f0f0")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IndustrialRobotGUI()
    window.show()
    sys.exit(app.exec())