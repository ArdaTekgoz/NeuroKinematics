from PyQt6.QtWidgets import QSplashScreen, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QFontDatabase, QPen
from PyQt6.QtSvg import QSvgRenderer
import os

class NeuroSplash(QSplashScreen):
    def __init__(self):
        # Splash screen boyutu
        width, height = 800, 600
        
        # Boş pixmap oluştur
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(0, 0, 0))
        
        super().__init__(pixmap)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Garet font yükle (sistem fontlarına eklemek için)
        self.load_custom_font()
        
        # Widget container
        container = QWidget(self)
        container.setGeometry(0, 0, width, height)
        container.setStyleSheet("background: transparent;")
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(60, 80, 60, 80)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Logo alanı (SVG veya PNG için placeholder)
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setFixedHeight(280)
        self.logo_label.setStyleSheet("background: transparent;")
        
        # Logo görselini yükle
        self.load_logo()
        
        layout.addWidget(self.logo_label)
        
        # Ana Başlık - NeuroKinematics
        self.title = QLabel("NeuroKinematics")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Garet", 48, QFont.Weight.Bold)
        if not QFontDatabase.families().__contains__("Garet"):
            title_font = QFont("Segoe UI", 48, QFont.Weight.Bold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        self.title.setFont(title_font)
        self.title.setStyleSheet("""
            QLabel {
                background: transparent;
                color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #14ffec,
                    stop: 0.5 #3d8ef7,
                    stop: 1 #14ffec
                );
            }
        """)
        layout.addWidget(self.title)
        
        # Spacer
        layout.addSpacing(30)
        
        # İmza - Designed by
        self.signature = QLabel("Designed and developed by Arda Tekgöz.")
        self.signature.setAlignment(Qt.AlignmentFlag.AlignCenter)
        signature_font = QFont("Garet", 13, QFont.Weight.Normal)
        if not QFontDatabase.families().__contains__("Garet"):
            signature_font = QFont("Segoe UI", 13, QFont.Weight.Normal)
        signature_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        self.signature.setFont(signature_font)
        self.signature.setStyleSheet("""
            QLabel {
                background: transparent;
                color: #ffffff;
                padding: 8px 20px;
            }
        """)
        layout.addWidget(self.signature)
        
        layout.addSpacing(20)
        
        # Loading mesajı için label
        self.loading_label = QLabel("")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_font = QFont("Garet", 11)
        if not QFontDatabase.families().__contains__("Garet"):
            loading_font = QFont("Segoe UI", 11)
        self.loading_label.setFont(loading_font)
        self.loading_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: #808080;
                font-style: italic;
            }
        """)
        layout.addWidget(self.loading_label)
        
        # Fade-in animasyonu için opacity
        self.setWindowOpacity(0.0)
        self.fade_in_animation()
    
    def load_custom_font(self):
        """Garet fontunu yükle"""
        # Garet font dosyaları varsa yükle
        font_paths = [
            "fonts/Garet-Book.ttf",
            "fonts/Garet-Heavy.ttf",
            "./Garet-Book.ttf",
            "./Garet-Heavy.ttf"
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                QFontDatabase.addApplicationFont(font_path)
    
    def load_logo(self):
        """Logo görselini yükle"""
        logo_paths = [
            "../assets/logo.png",  # From src/ directory
            "../assets/logo.svg",  # From src/ directory
            "assets/logo.png",     # From project root
            "assets/logo.svg",     # From project root
            "logo.png",
            "logo.svg"
        ]
        
        logo_loaded = False
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                if logo_path.endswith('.svg'):
                    # SVG için renderer kullan
                    renderer = QSvgRenderer(logo_path)
                    pixmap = QPixmap(400, 280)
                    pixmap.fill(Qt.GlobalColor.transparent)
                    painter = QPainter(pixmap)
                    renderer.render(painter)
                    painter.end()
                    self.logo_label.setPixmap(pixmap)
                else:
                    # PNG için doğrudan yükle
                    pixmap = QPixmap(logo_path)
                    scaled_pixmap = pixmap.scaled(
                        400, 280,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.logo_label.setPixmap(scaled_pixmap)
                logo_loaded = True
                break
        
        if not logo_loaded:
            # Logo bulunamazsa, metin logosu göster
            self.logo_label.setText("🤖")
            self.logo_label.setStyleSheet("""
                QLabel {
                    background: transparent;
                    color: #14ffec;
                    font-size: 120px;
                }
            """)
    
    def fade_in_animation(self):
        """Fade-in animasyonu"""
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(1000)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.start()
    
    def showMessage(self, message, alignment=Qt.AlignmentFlag.AlignCenter, color=QColor(20, 255, 236)):
        """Loading mesajını göster"""
        # Gelen renk Qt.GlobalColor da olabilir, bunu her zaman QColor'a çevir
        qcolor = color if isinstance(color, QColor) else QColor(color)
        
        self.loading_label.setText(message)
        self.loading_label.setStyleSheet(f"""
            QLabel {{
                background: transparent;
                color: rgb({qcolor.red()}, {qcolor.green()}, {qcolor.blue()});
                font-style: italic;
                font-weight: 400;
            }}
        """)
        super().showMessage(message, alignment, qcolor)
    
    def paintEvent(self, event):
        """Custom paint event - siyah arka plan"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Siyah arka plan
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # İnce cyan border (opsiyonel)
        pen = QPen(QColor(20, 255, 236, 100))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(1, 1, self.width()-2, self.height()-2)
        
        painter.end()
        super().paintEvent(event)
