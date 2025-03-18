from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QTabWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPainter, QPainterPath
from utils.D2_extraccion import TabD2
from utils.D3_Matching import TabD3
from utils.D9_lines import TabD9


# =======================================================
#               PESTAÑA "Acerca de"
# =======================================================
class TabAbout(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        image_label = QLabel(self)
        pixmap = QPixmap("img/Foto.jpg")
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                200,
                200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            circular_pixmap = QPixmap(pixmap.size())
            circular_pixmap.fill(Qt.GlobalColor.transparent)

            painter = QPainter(circular_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            path = QPainterPath()
            path.addEllipse(0, 0, pixmap.width(), pixmap.height())

            painter.setClipPath(path)

            painter.drawPixmap(0, 0, pixmap)
            painter.end()

            image_label.setPixmap(circular_pixmap)

        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Etiqueta de encabezado
        header_label = QLabel("Programa realizado por...", self)
        header_label.setStyleSheet("font-size: 14px; color: #333;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        # Autor
        author_label = QLabel("Junior Antonio Calvo Montañez", self)
        author_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(author_label)

        # Descripción
        description_label = QLabel("Web GIS | Remote Sensing | Programming", self)
        description_label.setStyleSheet("font-size: 12px; color: #555;")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description_label)

        # Ubicación
        location_label = QLabel("Perú", self)
        location_label.setStyleSheet("font-size: 12px; color: #777; font-weight: bold;")
        location_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(location_label)

        # Universidad
        university_label = QLabel("📍 Universidad de Salamanca", self)
        university_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #B71C1C;"
        )
        university_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(university_label)

        layout.addStretch()
        self.setLayout(layout)


# =======================================================
#               VENTANA PRINCIPAL
# =======================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desarrollo de Algoritmos PIA")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.tab_d2 = TabD2()
        self.tab_d3 = TabD3()
        self.tab_d9 = TabD9()
        self.tap_about = TabAbout()

        self.tabs.addTab(self.tab_d2, "D2: Extracción")
        self.tabs.addTab(self.tab_d3, "D3: Matching")
        self.tabs.addTab(self.tab_d9, "D9: Líneas")
        self.tabs.addTab(self.tap_about, "Acerca de")

        self.setCentralWidget(self.tabs)
