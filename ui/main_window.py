# ====== IMPORTS ======
import sys
import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QMenu, QFileDialog, QMessageBox, QWidget, QLineEdit, QGroupBox, QFormLayout, 
    QHeaderView, QDialog, QPushButton, QComboBox, QTabWidget, QTextEdit, QScrollArea
)
from PyQt6.QtGui import QAction, QPixmap, QImage, QPainter, QPainterPath
from PyQt6.QtCore import Qt

# ====== CLASE PRINCIPAL ======
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento de Imágenes: D2, D3 y D9")
        self.resize(1000, 700)

        # ====== Variables ======
        self.original_image = None  # Guardará la imagen original (cv2)
        self.processed_image = None # Guardará la imagen con resultados de cada proceso (cv2)

        # ====== Configuración de la UI ======
        self.setup_ui()
        self.setup_menu()

    # --------------------------------------------------
    #  CONFIGURACIÓN PRINCIPAL DE LA UI
    # --------------------------------------------------
    def setup_ui(self):
        """
        Crea toda la interfaz, copiando el estilo CSS de tu ejemplo.
        """
        # Layout principal
        main_layout = QVBoxLayout()
        
        # -------- Título principal --------
        title_label = QLabel("Procesamiento de Imágenes")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px; 
                font-weight: bold; 
                color: #4CAF50; 
                margin-bottom: 10px;
            }
        """)
        main_layout.addWidget(title_label)

        # -------- Sección de Tabs --------
        # Usaremos un QTabWidget para agrupar las tareas D2, D3, D9
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { background-color: #ffffff; }")
        
        # Pestaña 1: D2 - Extracción de Características
        self.tab_d2 = QWidget()
        self.setup_tab_d2()
        
        # Pestaña 2: D3 - Matching de Características
        self.tab_d3 = QWidget()
        self.setup_tab_d3()
        
        # Pestaña 3: D9 - Matching de Líneas
        self.tab_d9 = QWidget()
        self.setup_tab_d9()

        # Agregar pestañas al QTabWidget
        self.tabs.addTab(self.tab_d2, "D2: Extracción")
        self.tabs.addTab(self.tab_d3, "D3: Matching")
        self.tabs.addTab(self.tab_d9, "D9: Matching Líneas")

        main_layout.addWidget(self.tabs)

        # -------- Contenedor principal --------
        container = QWidget()
        container.setLayout(main_layout)
        container.setStyleSheet("QWidget { background-color: #ffffff; }")
        self.setCentralWidget(container)

    # --------------------------------------------------
    
    #  CONFIGURACIÓN DE LA PESTAÑA D2 (Extracción)
    def setup_tab_d2(self):
        layout = QVBoxLayout()

        # Grupo para la imagen
        self.group_image_d2 = QGroupBox("Imagen Original / Procesada")
        self.group_image_d2.setStyleSheet("""
            QGroupBox {
                font-size: 18px; 
                font-weight: bold; 
                color: #4CAF50;
                margin-top: 10px; 
                padding: 10px; 
                border: 1px solid #ddd; 
                border-radius: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 10px; 
            }
        """)
        image_layout = QHBoxLayout()

        # Creamos el QLabel para la imagen
        self.label_image_d2 = QLabel("No hay imagen cargada.")
        self.label_image_d2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_image_d2.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0; 
                border: 1px solid #ccc; 
                font-size: 14px;
            }
        """)

        # Opción 1: que el QLabel se adapte al tamaño real de la imagen (sin forzar)
        #    => Se mostrará la imagen en su tamaño real y se verán scrollbars si es muy grande
        # self.label_image_d2.setScaledContents(False)

        # Opción 2: forzar que la imagen siempre ocupe el tamaño del label
        #    => Se reescala la imagen a la medida del label, perdiendo detalle si es muy grande
        self.label_image_d2.setScaledContents(True)

        # Creamos el QScrollArea y metemos el label
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.label_image_d2)

        # Agregamos el QScrollArea al layout
        image_layout.addWidget(scroll_area)
        self.group_image_d2.setLayout(image_layout)

        # Grupo para la configuración de extracción (igual que antes)
        self.group_extraction = QGroupBox("Configuración de Extracción")
        ...
        # resto de tu código
        # Tabla para mostrar resultados (keypoints, etc.)
        self.table_d2 = QTableWidget(0, 2)
        self.table_d2.setHorizontalHeaderLabels(["Keypoint", "Valor"])
        self.table_d2.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_d2.setStyleSheet("""
            QTableWidget {
                font-size: 14px; 
                background-color: #ffffff; 
                border: none;
            }
            QHeaderView::section {
                background-color: #4CAF50; 
                color: white; 
                font-size: 14px; 
                font-weight: bold; 
                padding: 5px;
            }
        """)

        layout.addWidget(self.group_image_d2)
        layout.addWidget(self.group_extraction)
        layout.addWidget(self.table_d2)

        self.tab_d2.setLayout(layout)
    # --------------------------------------------------
    #  CONFIGURACIÓN DE LA PESTAÑA D3 (Matching)
    # --------------------------------------------------
    def setup_tab_d3(self):
        """
        Pestaña para D3: Matching de características entre dos imágenes.
        (Podrías adaptar para cargar 2 imágenes, etc.)
        """
        layout = QVBoxLayout()

        # Grupo para la imagen
        self.group_image_d3 = QGroupBox("Imagen/Correspondencia")
        self.group_image_d3.setStyleSheet("""
            QGroupBox {
                font-size: 18px; 
                font-weight: bold; 
                color: #4CAF50;
                margin-top: 10px; 
                padding: 10px; 
                border: 1px solid #ddd; 
                border-radius: 10px;
                background-color: #ffffff;
            }
        """)
        image_layout = QHBoxLayout()

        self.label_image_d3 = QLabel("No hay imagen cargada.")
        self.label_image_d3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_image_d3.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0; 
                border: 1px solid #ccc; 
                font-size: 14px;
            }
        """)
        image_layout.addWidget(self.label_image_d3)
        self.group_image_d3.setLayout(image_layout)

        # Grupo de configuración
        self.group_matching = QGroupBox("Configuración de Matching")
        self.group_matching.setStyleSheet("""
            QGroupBox {
                font-size: 18px; 
                font-weight: bold; 
                color: #4CAF50;
                margin-top: 10px; 
                padding: 10px; 
                border: 1px solid #ddd; 
                border-radius: 10px;
                background-color: #ffffff;
            }
        """)
        matching_layout = QHBoxLayout()

        # Combobox de algoritmo de matching (BF, FLANN, etc.)
        self.combo_matching = QComboBox()
        self.combo_matching.addItems(["Brute-Force", "FLANN"])
        self.combo_matching.setStyleSheet("""
            QComboBox {
                font-size: 14px; 
                padding: 5px; 
                background-color: #ffffff; 
                border: 1px solid #ccc; 
                border-radius: 5px;
            }
        """)

        # Botón para ejecutar matching
        self.btn_match = QPushButton("Ejecutar Matching")
        self.btn_match.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                border: none; 
                border-radius: 5px; 
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.btn_match.clicked.connect(self.match_features)

        matching_layout.addWidget(QLabel("Método:"))
        matching_layout.addWidget(self.combo_matching)
        matching_layout.addWidget(self.btn_match)
        self.group_matching.setLayout(matching_layout)

        # Área de texto o tabla para mostrar resultados del matching
        self.text_d3 = QTextEdit()
        self.text_d3.setStyleSheet("""
            QTextEdit {
                font-size: 14px;
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)

        layout.addWidget(self.group_image_d3)
        layout.addWidget(self.group_matching)
        layout.addWidget(self.text_d3)
        self.tab_d3.setLayout(layout)

    # --------------------------------------------------
    #  CONFIGURACIÓN DE LA PESTAÑA D9 (Matching de Líneas)
    # --------------------------------------------------
    def setup_tab_d9(self):
        """
        Pestaña para D9: Matching de líneas. 
        Aquí podrías implementar la detección de líneas (por ejemplo, HoughLines) 
        y luego un matching entre imágenes, etc.
        """
        layout = QVBoxLayout()

        # Grupo para la imagen
        self.group_image_d9 = QGroupBox("Imagen/Matching de Líneas")
        self.group_image_d9.setStyleSheet("""
            QGroupBox {
                font-size: 18px; 
                font-weight: bold; 
                color: #4CAF50;
                margin-top: 10px; 
                padding: 10px; 
                border: 1px solid #ddd; 
                border-radius: 10px;
                background-color: #ffffff;
            }
        """)
        image_layout = QHBoxLayout()

        self.label_image_d9 = QLabel("No hay imagen cargada.")
        self.label_image_d9.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_image_d9.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0; 
                border: 1px solid #ccc; 
                font-size: 14px;
            }
        """)
        image_layout.addWidget(self.label_image_d9)
        self.group_image_d9.setLayout(image_layout)

        # Grupo de configuración
        self.group_lines = QGroupBox("Configuración de Matching de Líneas")
        self.group_lines.setStyleSheet("""
            QGroupBox {
                font-size: 18px; 
                font-weight: bold; 
                color: #4CAF50;
                margin-top: 10px; 
                padding: 10px; 
                border: 1px solid #ddd; 
                border-radius: 10px;
                background-color: #ffffff;
            }
        """)
        lines_layout = QHBoxLayout()

        # Botón para ejecutar el matching de líneas
        self.btn_lines = QPushButton("Ejecutar Matching de Líneas")
        self.btn_lines.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                border: none; 
                border-radius: 5px; 
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.btn_lines.clicked.connect(self.match_lines)

        lines_layout.addWidget(self.btn_lines)
        self.group_lines.setLayout(lines_layout)

        # Texto para mostrar resultados
        self.text_d9 = QTextEdit()
        self.text_d9.setStyleSheet("""
            QTextEdit {
                font-size: 14px;
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)

        layout.addWidget(self.group_image_d9)
        layout.addWidget(self.group_lines)
        layout.addWidget(self.text_d9)
        self.tab_d9.setLayout(layout)

    # --------------------------------------------------
    #  MENÚ Y ACCIONES
    # --------------------------------------------------
    def setup_menu(self):
        menu_bar = self.menuBar()
        
        # ====== Estilos ======
        style_bar = """
            QMenuBar {
                background-color: #E0E0E0;  /* Fondo gris claro */
                color: black;
                font-size: 14px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 12px;
                margin: 2px;
                border-radius: 6px;
            }
            QMenuBar::item:selected { 
                background-color: #BDBDBD; 
                color: black;
                border-radius: 6px;
            }
            QMenu {
                background-color: #F5F5F5;
                border: 1px solid #D0D0D0;
                border-radius: 6px;
                padding: 5px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 8px 20px;
                color: black;
                font-size: 14px;
                margin: 2px;
                border-radius: 4px;
            }
            QMenu::item:selected { 
                background-color: #BDBDBD;
                color: black;
                border: 1px solid #A0A0A0;
                border-radius: 4px;
            }
            QMenu::separator {
                height: 1px;
                background: #D0D0D0;
                margin: 5px 10px;
            }
        """
        menu_bar.setStyleSheet(style_bar)

        # Menús
        file_menu = QMenu("Archivo", self)
        help_menu = QMenu("Ayuda", self)

        # Acciones
        open_image_action = QAction("Abrir Imagen", self)
        exit_action = QAction("Salir", self)
        about_action = QAction("Acerca de", self)

        # Conexiones
        open_image_action.triggered.connect(self.open_image)
        exit_action.triggered.connect(self.close)
        about_action.triggered.connect(self.show_about)

        # Agregar acciones
        file_menu.addAction(open_image_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        help_menu.addAction(about_action)

        # Agregar menús a la barra
        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(help_menu)

    # --------------------------------------------------
    #  FUNCIONES PRINCIPALES
    # --------------------------------------------------
    def open_image(self):
        """
        Abre un archivo de imagen y la muestra en la pestaña actual.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagen", "", "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            # Cargar imagen con OpenCV
            self.original_image = cv2.imread(file_path)

            # Convertir a QPixmap para mostrar en QLabel
            self.display_image(self.original_image, self.label_image_d2)
            self.display_image(self.original_image, self.label_image_d3)
            self.display_image(self.original_image, self.label_image_d9)

            QMessageBox.information(self, "Imagen Cargada", f"Se ha cargado la imagen:\n{file_path}")

    def display_image(self, cv_img, label):
        """
        Convierte una imagen de OpenCV (BGR) a QPixmap y la muestra en un QLabel.
        """
        if cv_img is None:
            return
        # Convertir de BGR a RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # Ajustar pixmap al tamaño del label (manteniendo proporción)
        scaled_pixmap = pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)

    # ---------------- D2: Extracción de Características ----------------
    def extract_features(self):
        """
        Lógica para D2: Extraer características (SIFT, ORB, FAST, Harris, etc.)
        """
        if self.original_image is None:
            QMessageBox.warning(self, "Atención", "Primero carga una imagen.")
            return

        selected_algo = self.combo_detector.currentText()
        # Aquí podrías implementar la lógica para cada algoritmo
        # Ejemplo de ORB:
        # if selected_algo == "ORB":
        #     orb = cv2.ORB_create()
        #     kp, des = orb.detectAndCompute(self.original_image, None)
        #     ...
        #
        # Actualizar tabla_d2 con la información de los keypoints o estadísticas.

        # Solo como demostración, completamos la tabla con datos ficticios:
        self.table_d2.setRowCount(0)
        # Suponiendo que detectamos N keypoints
        dummy_keypoints = [("KP1", "Info1"), ("KP2", "Info2"), ("KP3", "Info3")]
        for (k, v) in dummy_keypoints:
            row = self.table_d2.rowCount()
            self.table_d2.insertRow(row)
            self.table_d2.setItem(row, 0, QTableWidgetItem(str(k)))
            self.table_d2.setItem(row, 1, QTableWidgetItem(str(v)))

        QMessageBox.information(self, "Extracción", f"Extracción realizada con {selected_algo} (demo).")

    # ---------------- D3: Matching de Características ----------------
    def match_features(self):
        """
        Lógica para D3: Matching de características (Brute Force, FLANN, etc.).
        """
        if self.original_image is None:
            QMessageBox.warning(self, "Atención", "Primero carga una imagen.")
            return

        selected_method = self.combo_matching.currentText()
        # Aquí la lógica real para matching entre 2 imágenes
        # (en este ejemplo, solo tenemos 1 imagen cargada, pero podrías ampliar a 2)

        # Ejemplo de demostración
        self.text_d3.clear()
        self.text_d3.append(f"Matching realizado con: {selected_method} (demo).")
        self.text_d3.append("Resultados ficticios: \n- Coincidencias encontradas: 123\n- Distancias promedio: 0.45")

    # ---------------- D9: Matching de Líneas ----------------
    def match_lines(self):
        """
        Lógica para D9: Matching de líneas (por ejemplo, detección de líneas con Hough 
        y luego comparación entre dos imágenes).
        """
        if self.original_image is None:
            QMessageBox.warning(self, "Atención", "Primero carga una imagen.")
            return

        # Ejemplo de demostración
        self.text_d9.clear()
        self.text_d9.append("Matching de líneas ejecutado (demo).")
        self.text_d9.append("Líneas detectadas: 5\nCoincidencias con otra imagen: 3")

    # --------------------------------------------------
    #  DIÁLOGO "ACERCA DE"
    # --------------------------------------------------
    def show_about(self):
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("Acerca de este programa")
        about_dialog.setFixedSize(400, 300)

        layout = QVBoxLayout()

        # Etiquetas de información
        label_info = QLabel("Ejemplo de interfaz PyQt6 para D2, D3 y D9.\n"
                            "Basado en la estructura y estilo CSS solicitados.")
        label_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_ok = QPushButton("Aceptar")
        btn_ok.clicked.connect(about_dialog.accept)
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                border: none; 
                border-radius: 5px;
                width: 80px; 
                height: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        layout.addStretch()
        layout.addWidget(label_info)
        layout.addStretch()
        layout.addWidget(btn_ok, alignment=Qt.AlignmentFlag.AlignCenter)

        about_dialog.setLayout(layout)
        about_dialog.exec()
