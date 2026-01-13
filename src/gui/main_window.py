import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSlider,
    QLabel,
    QFrame,
    QMenuBar,
    QMenu,
    QProgressBar,
    QGroupBox,
    QLineEdit,
    QComboBox,
    QRadioButton,
    QDial,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction


class IndustrialRobotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroKinematics - NextGen Industrial Robot Control System")
        self.setGeometry(100, 100, 1400, 800)
        self.current_language = "en"
        self.current_euler_convention = "Z-Y-X"
        self.tcp_trace_enabled = False
        
        # Çeviri sözlükleri
        self.translations = {
            "en": {
                "window_title": "NeuroKinematics - NextGen Industrial Robot Control System",
                "robot_controls": "Robot Controls",
                "joint": "Joint",
                "reference_frames": "Reference Frames",
                "world_frame": "World frame (w.r.t. ref. 0)",
                "tool_frame": "Tool frame (w.r.t. ref. 6)",
                "show_ref_frames": "Show reference frames",
                "world": "World",
                "tool": "Tool",
                "all_none": "All/none",
                "cartesian_jog": "Cartesian Jog",
                "position": "Position (tool frame w.r.t. world frame)",
                "orientation": "Orientation (tool frame w.r.t. world frame)",
                "euler_angles": "Euler angles",
                "quaternions": "Quaternions",
                "translation_along": "Translation along",
                "rotation_about": "Rotation about",
                "view": "View",
                "tcp_trace": "TCP Trace",
                "on": "ON",
                "off": "OFF",
                "file": "File",
                "load_robot": "Load a robot",
                "load_tool": "Load a tool",
                "remove_tool": "Remove the tool",
                "load_object": "Load object",
                "load_station": "Load station",
                "save_station": "Save station",
                "save_default_station": "Save as default station",
                "load_simulation": "Load a simulation",
                "exit": "Exit",
                "view_menu": "View",
                "robot_panel": "Robot control panel",
                "objects_panel": "Objects panel",
                "simulation_panel": "Simulation panel",
                "abb_program_data": "ABB program data",
                "joystick": "Joystick",
                "enable_joystick": "Enable joystick control",
                "options": "Options",
                "language": "Language",
                "euler_convention": "Euler angle convention",
                "tcp_trace_menu": "TCP trace",
                "enable_tcp": "Enable TCP trace",
                "help": "Help",
                "documentation": "Documentation",
                "about": "About NeuroKinematics",
                "isometric": "Isometric",
                "top": "Top",
                "front": "Front",
                "right": "Right",
                "left": "Left",
                "back": "Back",
            },
            "tr": {
                "window_title": "NeuroKinematics - Yeni Nesil Endüstriyel Robot Kontrol Sistemi",
                "robot_controls": "Robot Kontrolleri",
                "joint": "Eklem",
                "reference_frames": "Referans Çerçeveleri",
                "world_frame": "Dünya çerçevesi (ref. 0'a göre)",
                "tool_frame": "Alet çerçevesi (ref. 6'ya göre)",
                "show_ref_frames": "Referans çerçevelerini göster",
                "world": "Dünya",
                "tool": "Alet",
                "all_none": "Tümü/Hiçbiri",
                "cartesian_jog": "Kartezyen Jog",
                "position": "Konum (alet çerçevesi dünya çerçevesine göre)",
                "orientation": "Yönelim (alet çerçevesi dünya çerçevesine göre)",
                "euler_angles": "Euler açıları",
                "quaternions": "Kuaterniyonlar",
                "translation_along": "Öteleme",
                "rotation_about": "Dönüş",
                "view": "Görünüm",
                "tcp_trace": "TCP İzleme",
                "on": "AÇIK",
                "off": "KAPALI",
                "file": "Dosya",
                "load_robot": "Robot yükle",
                "load_tool": "Alet yükle",
                "remove_tool": "Aleti kaldır",
                "load_object": "Nesne yükle",
                "load_station": "İstasyon yükle",
                "save_station": "İstasyonu kaydet",
                "save_default_station": "Varsayılan istasyon olarak kaydet",
                "load_simulation": "Simülasyon yükle",
                "exit": "Çıkış",
                "view_menu": "Görünüm",
                "robot_panel": "Robot kontrol paneli",
                "objects_panel": "Nesneler paneli",
                "simulation_panel": "Simülasyon paneli",
                "abb_program_data": "ABB program verisi",
                "joystick": "Joystick",
                "enable_joystick": "Joystick kontrolünü etkinleştir",
                "options": "Seçenekler",
                "language": "Dil",
                "euler_convention": "Euler açı konvansiyonu",
                "tcp_trace_menu": "TCP izleme",
                "enable_tcp": "TCP izlemeyi etkinleştir",
                "help": "Yardım",
                "documentation": "Dokümantasyon",
                "about": "NeuroKinematics Hakkında",
                "isometric": "İzometrik",
                "top": "Üstten",
                "front": "Önden",
                "right": "Sağdan",
                "left": "Soldan",
                "back": "Arkadan",
            }
        }
        
        # Kamera görünüm isimleri
        self.camera_names = {
            "en": ["Isometric", "Top", "Front", "Right", "Left", "Back"],
            "tr": ["İzometrik", "Üstten", "Önden", "Sağdan", "Soldan", "Arkadan"]
        }
        
        # Ana widget ve layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sol Panel - Robot Kontrolleri
        self.left_panel = self.create_left_panel()
        main_layout.addWidget(self.left_panel)
        
        # Orta Panel - 3D Viewport
        self.center_panel = self.create_center_panel()
        main_layout.addWidget(self.center_panel, 1)
        
        # Sağ Panel - Reference Frames & Cartesian Jog
        self.right_panel = self.create_right_panel()
        main_layout.addWidget(self.right_panel)
        
        # Menu bar oluştur
        self.create_menu_bar()
        
        # Stylesheet uygula
        self.apply_stylesheet()
    
    def create_left_panel(self):
        """Sol panel - Robot eklem kontrolleri"""
        panel = QFrame()
        panel.setObjectName("leftPanel")
        panel.setFixedWidth(300)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Başlık
        self.left_panel_title = QLabel("Robot Controls")
        self.left_panel_title.setObjectName("panelTitle")
        self.left_panel_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.left_panel_title)
        
        # 6 eklem için slider'lar
        self.joint_sliders = []
        self.joint_labels = []
        self.joint_name_labels = []
        
        for i in range(1, 7):
            joint_container = QWidget()
            joint_layout = QVBoxLayout(joint_container)
            joint_layout.setContentsMargins(0, 10, 0, 10)
            joint_layout.setSpacing(5)
            
            # Eklem ismi ve değer
            header_layout = QHBoxLayout()
            joint_name = QLabel(f"Joint {i} (J{i})")
            joint_name.setObjectName("jointLabel")
            self.joint_name_labels.append(joint_name)
            
            value_label = QLabel("0°")
            value_label.setObjectName("valueLabel")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.joint_labels.append(value_label)
            
            header_layout.addWidget(joint_name)
            header_layout.addWidget(value_label)
            joint_layout.addLayout(header_layout)
            
            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setObjectName("jointSlider")
            slider.setMinimum(-180)
            slider.setMaximum(180)
            slider.setValue(0)
            slider.valueChanged.connect(
                lambda val, idx=i-1: self.update_joint_value(idx, val)
            )
            self.joint_sliders.append(slider)
            joint_layout.addWidget(slider)
            
            layout.addWidget(joint_container)
        
        layout.addStretch()
        return panel
    
    def create_center_panel(self):
        """Orta panel - 3D görüntüleme alanı"""
        panel = QFrame()
        panel.setObjectName("centerPanel")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 3D viewport placeholder
        viewport = QFrame()
        viewport.setObjectName("viewport3D")
        layout.addWidget(viewport)
        
        return panel
    
    def create_right_panel(self):
        """Sağ panel - Reference Frames & Cartesian Jog"""
        panel = QFrame()
        panel.setObjectName("rightPanel")
        panel.setFixedWidth(350)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Başlık
        self.right_panel_title = QLabel("Kinematics Control")
        self.right_panel_title.setObjectName("panelTitle")
        self.right_panel_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.right_panel_title)

        # Reference Frames Group
        self.ref_group = QGroupBox("Reference Frames")
        self.ref_group.setObjectName("refGroup")
        ref_layout = QGridLayout(self.ref_group)
        ref_layout.setContentsMargins(10, 10, 10, 10)
        ref_layout.setHorizontalSpacing(6)
        ref_layout.setVerticalSpacing(4)

        # World frame
        self.world_frame_label = QLabel("World frame (w.r.t. ref. 0)")
        self.world_frame_edit = QLineEdit("0.00 mm, 0.00 mm, 0.00 mm, 0.00°, 0.00°, 0.00°")
        self.world_frame_edit.setReadOnly(True)
        ref_layout.addWidget(self.world_frame_label, 0, 0)
        ref_layout.addWidget(self.world_frame_edit, 1, 0, 1, 4)

        # Tool frame
        self.tool_frame_label = QLabel("Tool frame (w.r.t. ref. 6)")
        self.tool_frame_edit = QLineEdit("1.6 mm, -82.20 mm, 101.79 mm, 90.00°, -60.00°, 0.00°")
        self.tool_frame_edit.setReadOnly(True)
        ref_layout.addWidget(self.tool_frame_label, 2, 0)
        ref_layout.addWidget(self.tool_frame_edit, 3, 0, 1, 4)

        # Show reference frames
        self.show_ref_frames_label = QLabel("Show reference frames")
        ref_layout.addWidget(self.show_ref_frames_label, 4, 0, 1, 2)

        self.world_frame_radio = QRadioButton("World")
        self.tool_frame_radio = QRadioButton("Tool")
        self.all_frame_radio = QRadioButton("All/none")
        self.world_frame_radio.setChecked(True)
        ref_layout.addWidget(self.world_frame_radio, 5, 0)
        ref_layout.addWidget(self.tool_frame_radio, 5, 1)
        ref_layout.addWidget(self.all_frame_radio, 5, 2)

        layout.addWidget(self.ref_group)

        # Cartesian Jog Group
        self.cart_group = QGroupBox("Cartesian Jog")
        self.cart_group.setObjectName("cartGroup")
        cart_layout = QVBoxLayout(self.cart_group)
        cart_layout.setContentsMargins(10, 10, 10, 10)
        cart_layout.setSpacing(8)

        # Position
        self.pos_label = QLabel("Position (tool frame w.r.t. world frame)")
        self.cart_position_edit = QLineEdit("475.785 mm, -82.198 mm, 628.840 mm")
        cart_layout.addWidget(self.pos_label)
        cart_layout.addWidget(self.cart_position_edit)

        # Orientation - euler & quaternion
        self.orient_label = QLabel("Orientation (tool frame w.r.t. world frame)")
        cart_layout.addWidget(self.orient_label)

        self.euler_label = QLabel("Euler angles ({}):".format(self.current_euler_convention))
        self.euler_edit = QLineEdit("30.000°, 0.000°, 90.000°")
        cart_layout.addWidget(self.euler_label)
        cart_layout.addWidget(self.euler_edit)

        self.quat_label = QLabel("Quaternions:")
        self.quat_edit = QLineEdit("0.68301, 0.68301, 0.18301, 0.18301")
        cart_layout.addWidget(self.quat_label)
        cart_layout.addWidget(self.quat_edit)

        # Translation / Rotation radio buttons
        motion_row = QHBoxLayout()
        self.motion_label = QLabel("Translation along")
        self.tx_radio = QRadioButton("X")
        self.ty_radio = QRadioButton("Y")
        self.tz_radio = QRadioButton("Z")
        self.tx_radio.setChecked(True)
        motion_row.addWidget(self.motion_label)
        motion_row.addWidget(self.tx_radio)
        motion_row.addWidget(self.ty_radio)
        motion_row.addWidget(self.tz_radio)
        motion_row.addStretch()
        cart_layout.addLayout(motion_row)

        rot_row = QHBoxLayout()
        self.rot_label = QLabel("Rotation about")
        self.rx_radio = QRadioButton("X")
        self.ry_radio = QRadioButton("Y")
        self.rz_radio = QRadioButton("Z")
        self.rz_radio.setChecked(True)
        rot_row.addWidget(self.rot_label)
        rot_row.addWidget(self.rx_radio)
        rot_row.addWidget(self.ry_radio)
        rot_row.addWidget(self.rz_radio)
        rot_row.addStretch()
        cart_layout.addLayout(rot_row)

        # Jog dial
        dial_row = QHBoxLayout()
        self.jog_dial = QDial()
        self.jog_dial.setMinimum(-180)
        self.jog_dial.setMaximum(180)
        self.jog_dial.setValue(0)
        self.jog_dial.valueChanged.connect(self.handle_cartesian_jog)
        dial_row.addStretch()
        dial_row.addWidget(self.jog_dial)
        dial_row.addStretch()
        cart_layout.addLayout(dial_row)

        layout.addWidget(self.cart_group)

        # Status label for showing selected configuration / view
        self.status_label = QLabel("View: Isometric | Euler: {} | TCP Trace: OFF".format(self.current_euler_convention))
        self.status_label.setObjectName("metricLabel")
        layout.addWidget(self.status_label)

        layout.addStretch()
        return panel
    
    def create_menu_bar(self):
        """Menu bar - competitor style with working actions"""
        menubar = self.menuBar()

        # FILE MENU
        self.file_menu = menubar.addMenu("File")

        self.load_robot_action = QAction("Load a robot", self)
        self.load_robot_action.setShortcut("Ctrl+R")
        self.load_robot_action.triggered.connect(self.load_robot)
        self.file_menu.addAction(self.load_robot_action)

        self.load_tool_action = QAction("Load a tool", self)
        self.load_tool_action.setShortcut("Ctrl+T")
        self.load_tool_action.triggered.connect(self.load_tool)
        self.file_menu.addAction(self.load_tool_action)

        self.remove_tool_action = QAction("Remove the tool", self)
        self.remove_tool_action.triggered.connect(self.remove_tool)
        self.file_menu.addAction(self.remove_tool_action)

        self.file_menu.addSeparator()

        self.load_object_action = QAction("Load object", self)
        self.load_object_action.setShortcut("Ctrl+O")
        self.load_object_action.triggered.connect(self.load_object)
        self.file_menu.addAction(self.load_object_action)

        self.file_menu.addSeparator()

        self.load_station_action = QAction("Load station", self)
        self.load_station_action.setShortcut("Ctrl+S")
        self.load_station_action.triggered.connect(self.load_station)
        self.file_menu.addAction(self.load_station_action)

        self.save_station_action = QAction("Save station", self)
        self.save_station_action.setShortcut("Ctrl+Shift+S")
        self.save_station_action.triggered.connect(self.save_station)
        self.file_menu.addAction(self.save_station_action)

        self.save_default_station_action = QAction("Save as default station", self)
        self.save_default_station_action.triggered.connect(self.save_default_station)
        self.file_menu.addAction(self.save_default_station_action)

        self.file_menu.addSeparator()

        self.load_simulation_action = QAction("Load a simulation", self)
        self.load_simulation_action.setShortcut("Ctrl+D")
        self.load_simulation_action.triggered.connect(self.load_simulation)
        self.file_menu.addAction(self.load_simulation_action)

        self.file_menu.addSeparator()

        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut("Alt+F4")
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        # VIEW MENU
        self.view_menu = menubar.addMenu("View")

        self.robot_panel_action = QAction("Robot control panel", self, checkable=True)
        self.robot_panel_action.setShortcut("F2")
        self.robot_panel_action.setChecked(True)
        self.robot_panel_action.triggered.connect(self.toggle_robot_panel)
        self.view_menu.addAction(self.robot_panel_action)

        self.objects_panel_action = QAction("Objects panel", self, checkable=True)
        self.objects_panel_action.setShortcut("F3")
        self.objects_panel_action.setChecked(True)
        self.objects_panel_action.triggered.connect(self.toggle_objects_panel)
        self.view_menu.addAction(self.objects_panel_action)

        self.simulation_panel_action = QAction("Simulation panel", self, checkable=True)
        self.simulation_panel_action.setShortcut("F4")
        self.simulation_panel_action.setChecked(True)
        self.simulation_panel_action.triggered.connect(self.toggle_simulation_panel)
        self.view_menu.addAction(self.simulation_panel_action)

        self.abb_program_data_action = QAction("ABB program data", self, checkable=True)
        self.abb_program_data_action.setShortcut("F5")
        self.abb_program_data_action.setChecked(True)
        self.abb_program_data_action.triggered.connect(self.toggle_abb_program_panel)
        self.view_menu.addAction(self.abb_program_data_action)

        self.view_menu.addSeparator()

        # Camera views (change label only – placeholder for real 3D view)
        self.camera_view_actions = []
        # Initialize with English names, will be updated when language changes
        for idx, name in enumerate(self.camera_names["en"]):
            action = QAction(name, self)
            action.triggered.connect(lambda checked=False, i=idx: self.set_view_by_index(i))
            self.view_menu.addAction(action)
            self.camera_view_actions.append(action)

        # JOYSTICK MENU
        self.joystick_menu = menubar.addMenu("Joystick")
        self.joystick_enable_action = QAction("Enable joystick control", self, checkable=True)
        self.joystick_enable_action.triggered.connect(self.toggle_joystick)
        self.joystick_menu.addAction(self.joystick_enable_action)

        # OPTIONS MENU
        self.options_menu = menubar.addMenu("Options")

        # Language submenu
        self.lang_menu = self.options_menu.addMenu("Language")
        self.lang_en_action = QAction("English", self, checkable=True)
        self.lang_tr_action = QAction("Türkçe", self, checkable=True)
        self.lang_en_action.setChecked(True)
        self.lang_en_action.triggered.connect(lambda: self.set_language("en"))
        self.lang_tr_action.triggered.connect(lambda: self.set_language("tr"))
        self.lang_menu.addAction(self.lang_en_action)
        self.lang_menu.addAction(self.lang_tr_action)

        # Euler angle convention submenu
        self.euler_menu = self.options_menu.addMenu("Euler angle convention")
        self.euler_zyx_action = QAction("Z-Y-X", self, checkable=True)
        self.euler_xyz_action = QAction("X-Y-Z", self, checkable=True)
        self.euler_zyz_action = QAction("Z-Y-Z", self, checkable=True)
        self.euler_zyx_action.setChecked(True)
        self.euler_zyx_action.triggered.connect(lambda: self.set_euler_convention("Z-Y-X"))
        self.euler_xyz_action.triggered.connect(lambda: self.set_euler_convention("X-Y-Z"))
        self.euler_zyz_action.triggered.connect(lambda: self.set_euler_convention("Z-Y-Z"))
        self.euler_menu.addAction(self.euler_zyx_action)
        self.euler_menu.addAction(self.euler_xyz_action)
        self.euler_menu.addAction(self.euler_zyz_action)

        # TCP trace submenu
        self.tcp_menu = self.options_menu.addMenu("TCP trace")
        self.tcp_enable_action = QAction("Enable TCP trace", self, checkable=True)
        self.tcp_enable_action.triggered.connect(self.toggle_tcp_trace)
        self.tcp_menu.addAction(self.tcp_enable_action)

        # HELP MENU
        self.help_menu = menubar.addMenu("Help")
        self.docs_action = QAction("Documentation", self)
        self.docs_action.setShortcut("F1")
        self.docs_action.triggered.connect(self.show_docs)
        self.help_menu.addAction(self.docs_action)

        self.about_action = QAction("About NeuroKinematics", self)
        self.about_action.triggered.connect(self.show_about)
        self.help_menu.addAction(self.about_action)
    
    def update_joint_value(self, index, value):
        """Eklem değerini güncelle"""
        self.joint_labels[index].setText(f"{value}°")
    
    def reset_joints(self):
        """Tüm eklemi sıfırla"""
        for slider in self.joint_sliders:
            slider.setValue(0)
    
    def toggle_fullscreen(self, checked):
        """Tam ekran geçişi"""
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    # ===== File menu handlers =====
    def load_robot(self):
        QFileDialog.getOpenFileName(self, "Load robot", "", "Robot files (*.urdf *.xml);;All files (*.*)")

    def load_tool(self):
        QFileDialog.getOpenFileName(self, "Load tool", "", "Tool files (*.json *.yaml *.xml);;All files (*.*)")

    def remove_tool(self):
        QMessageBox.information(self, "Tool", "Tool removed from robot (placeholder).")

    def load_object(self):
        QFileDialog.getOpenFileName(self, "Load object", "", "Object files (*.stl *.obj *.step);;All files (*.*)")

    def load_station(self):
        QFileDialog.getOpenFileName(self, "Load station", "", "Station files (*.nkstation);;All files (*.*)")

    def save_station(self):
        QFileDialog.getSaveFileName(self, "Save station", "", "Station files (*.nkstation);;All files (*.*)")

    def save_default_station(self):
        QMessageBox.information(self, "Station", "Current station saved as default (placeholder).")

    def load_simulation(self):
        QFileDialog.getOpenFileName(self, "Load simulation", "", "Simulation files (*.nksim);;All files (*.*)")

    # ===== View menu handlers =====
    def toggle_robot_panel(self, checked: bool):
        self.left_panel.setVisible(checked)

    def toggle_objects_panel(self, checked: bool):
        # No separate objects panel yet – just acknowledge.
        QMessageBox.information(self, "Objects panel", "Objects panel toggled (not implemented yet).")

    def toggle_simulation_panel(self, checked: bool):
        QMessageBox.information(self, "Simulation panel", "Simulation panel toggled (not implemented yet).")

    def toggle_abb_program_panel(self, checked: bool):
        QMessageBox.information(self, "ABB program data", "ABB program data panel toggled (not implemented yet).")

    def set_view_by_index(self, index: int):
        """Set view by camera index (0=Isometric, 1=Top, etc.)"""
        t = self.translations[self.current_language]
        camera_names = self.camera_names[self.current_language]
        if 0 <= index < len(camera_names):
            view_text = camera_names[index]
            self.status_label.setText(
                f"{t['view']}: {view_text} | Euler: {self.current_euler_convention} | "
                f"{t['tcp_trace']}: {t['on' if self.tcp_trace_enabled else 'off']}"
            )
    
    def set_view(self, name: str):
        """Legacy function - kept for compatibility"""
        t = self.translations[self.current_language]
        camera_names = self.camera_names[self.current_language]
        try:
            idx = camera_names.index(name)
            view_text = camera_names[idx]
        except ValueError:
            # Fallback if name not found
            view_text = name
        self.status_label.setText(
            f"{t['view']}: {view_text} | Euler: {self.current_euler_convention} | "
            f"{t['tcp_trace']}: {t['on' if self.tcp_trace_enabled else 'off']}"
        )

    # ===== Joystick / options handlers =====
    def toggle_joystick(self, checked: bool):
        QMessageBox.information(
            self,
            "Joystick",
            "Joystick control {}".format("enabled" if checked else "disabled"),
        )

    def set_language(self, lang: str):
        self.current_language = lang
        # Update checked state manually for exclusivity
        self.lang_en_action.setChecked(lang == "en")
        self.lang_tr_action.setChecked(lang == "tr")
        # Update all UI texts
        self.update_ui_language()
    
    def update_ui_language(self):
        """Update all UI texts based on current language"""
        t = self.translations[self.current_language]
        
        # Window title
        self.setWindowTitle(t["window_title"])
        
        # Left panel
        self.left_panel_title.setText(t["robot_controls"])
        for i, joint_label in enumerate(self.joint_name_labels, 1):
            joint_label.setText(f"{t['joint']} {i} (J{i})")
        
        # Right panel
        self.right_panel_title.setText("Kinematics Control")  # Keep same for now
        
        # Reference Frames group
        self.ref_group.setTitle(t["reference_frames"])
        self.world_frame_label.setText(t["world_frame"])
        self.tool_frame_label.setText(t["tool_frame"])
        self.show_ref_frames_label.setText(t["show_ref_frames"])
        self.world_frame_radio.setText(t["world"])
        self.tool_frame_radio.setText(t["tool"])
        self.all_frame_radio.setText(t["all_none"])
        
        # Cartesian Jog group
        self.cart_group.setTitle(t["cartesian_jog"])
        self.pos_label.setText(t["position"])
        self.orient_label.setText(t["orientation"])
        self.euler_label.setText(f"{t['euler_angles']} ({self.current_euler_convention}):")
        self.quat_label.setText(f"{t['quaternions']}:")
        self.motion_label.setText(t["translation_along"])
        self.rot_label.setText(t["rotation_about"])
        
        # Status label
        view_text = self.camera_names[self.current_language][0]  # Isometric/İzometrik
        self.status_label.setText(
            f"{t['view']}: {view_text} | Euler: {self.current_euler_convention} | "
            f"{t['tcp_trace']}: {t['on' if self.tcp_trace_enabled else 'off']}"
        )
        
        # Menu bar titles
        self.file_menu.setTitle(t["file"])
        self.view_menu.setTitle(t["view_menu"])
        self.joystick_menu.setTitle(t["joystick"])
        self.options_menu.setTitle(t["options"])
        self.lang_menu.setTitle(t["language"])
        self.euler_menu.setTitle(t["euler_convention"])
        self.tcp_menu.setTitle(t["tcp_trace_menu"])
        self.help_menu.setTitle(t["help"])
        
        # Menu bar actions
        # File menu
        self.load_robot_action.setText(t["load_robot"])
        self.load_tool_action.setText(t["load_tool"])
        self.remove_tool_action.setText(t["remove_tool"])
        self.load_object_action.setText(t["load_object"])
        self.load_station_action.setText(t["load_station"])
        self.save_station_action.setText(t["save_station"])
        self.save_default_station_action.setText(t["save_default_station"])
        self.load_simulation_action.setText(t["load_simulation"])
        self.exit_action.setText(t["exit"])
        
        # View menu
        self.robot_panel_action.setText(t["robot_panel"])
        self.objects_panel_action.setText(t["objects_panel"])
        self.simulation_panel_action.setText(t["simulation_panel"])
        self.abb_program_data_action.setText(t["abb_program_data"])
        
        # Update camera view action names
        camera_names = self.camera_names[self.current_language]
        for i, action in enumerate(self.camera_view_actions):
            if i < len(camera_names):
                action.setText(camera_names[i])
        
        # Joystick menu
        self.joystick_enable_action.setText(t["enable_joystick"])
        
        # Options menu
        self.tcp_enable_action.setText(t["enable_tcp"])
        
        # Help menu
        self.docs_action.setText(t["documentation"])
        self.about_action.setText(t["about"])

    def set_euler_convention(self, convention: str):
        self.current_euler_convention = convention
        self.euler_zyx_action.setChecked(convention == "Z-Y-X")
        self.euler_xyz_action.setChecked(convention == "X-Y-Z")
        self.euler_zyz_action.setChecked(convention == "Z-Y-Z")
        t = self.translations[self.current_language]
        self.euler_label.setText(f"{t['euler_angles']} ({convention}):")
        # Refresh status label
        view_text = self.camera_names[self.current_language][0]
        self.status_label.setText(
            f"{t['view']}: {view_text} | Euler: {self.current_euler_convention} | "
            f"{t['tcp_trace']}: {t['on' if self.tcp_trace_enabled else 'off']}"
        )

    def toggle_tcp_trace(self, checked: bool):
        self.tcp_trace_enabled = checked
        t = self.translations[self.current_language]
        view_text = self.camera_names[self.current_language][0]
        self.status_label.setText(
            f"{t['view']}: {view_text} | Euler: {self.current_euler_convention} | "
            f"{t['tcp_trace']}: {t['on' if self.tcp_trace_enabled else 'off']}"
        )

    def show_docs(self):
        QMessageBox.information(self, "Documentation", "Open the NeuroKinematics documentation (placeholder).")

    def show_about(self):
        QMessageBox.information(
            self,
            "About NeuroKinematics",
            "NeuroKinematics _ Next-Gen Industrial Robot Control System\nGUI prototype.",
        )

    def handle_cartesian_jog(self, value: int):
        # Placeholder: simply reflect dial value in the Euler Z component
        parts = self.euler_edit.text().split(",")
        if len(parts) == 3:
            parts[2] = f" {value:.3f}°"
            self.euler_edit.setText(",".join(parts))
    
    def apply_stylesheet(self):
        """Modern endüstriyel stylesheet"""
        stylesheet = """
            /* Genel Ayarlar */
            QMainWindow {
                background-color: #101018;
            }
            
            QMenuBar {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 5px;
                border-bottom: 2px solid #3d3d3d;
            }
            
            QMenuBar::item {
                padding: 5px 12px;
                background-color: transparent;
                border-radius: 3px;
            }
            
            QMenuBar::item:selected {
                background-color: #0d7377;
            }
            
            QMenu {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                padding: 5px;
            }
            
            QMenu::item {
                padding: 8px 25px;
                border-radius: 3px;
            }
            
            QMenu::item:selected {
                background-color: #0d7377;
            }
            
            /* Panel Stilleri */
            QFrame#leftPanel, QFrame#rightPanel {
                background-color: #252525;
                border-right: 1px solid #3d3d3d;
            }
            
            QFrame#centerPanel {
                background-color: #101018;
            }
            
            QFrame#viewport3D {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #05070d,
                    stop: 0.4 #0b1733,
                    stop: 1 #121f2e
                );
                border: 2px solid #3d3d3d;
                border-radius: 5px;
            }
            
            /* Başlıklar */
            QLabel#panelTitle {
                font-size: 18px;
                font-weight: bold;
                color: #14ffec;
                padding: 10px;
                background-color: #2d2d2d;
                border-radius: 5px;
                border: 1px solid #0d7377;
            }
            
            /* Eklem Kontrolleri */
            QLabel#jointLabel {
                font-size: 13px;
                font-weight: 600;
                color: #b0b0b0;
            }
            
            QLabel#valueLabel {
                font-size: 14px;
                font-weight: bold;
                color: #14ffec;
                background-color: #2d2d2d;
                padding: 3px 10px;
                border-radius: 3px;
                min-width: 50px;
            }
            
            /* Slider Stilleri */
            QSlider#jointSlider::groove:horizontal {
                background-color: #3d3d3d;
                height: 8px;
                border-radius: 4px;
            }
            
            QSlider#jointSlider::handle:horizontal {
                background-color: #0d7377;
                border: 2px solid #14ffec;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            
            QSlider#jointSlider::handle:horizontal:hover {
                background-color: #14ffec;
            }
            
            QSlider#jointSlider::sub-page:horizontal {
                background-color: #0d7377;
                border-radius: 4px;
            }
            
            /* AI Metrics */
            QLabel#metricLabel {
                font-size: 12px;
                font-weight: 600;
                color: #a0a0a0;
                text-transform: uppercase;
            }
            
            QLabel#metricValue {
                font-size: 24px;
                font-weight: bold;
                color: #14ffec;
            }
            
            /* Progress Bar */
            QProgressBar#metricBar {
                background-color: #3d3d3d;
                border: none;
                border-radius: 5px;
                height: 10px;
                text-align: center;
            }
            
            QProgressBar#metricBar::chunk {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #0d7377,
                    stop: 1 #14ffec
                );
                border-radius: 5px;
            }
        """
        self.setStyleSheet(stylesheet)


def main():
    app = QApplication(sys.argv)
    window = IndustrialRobotGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()