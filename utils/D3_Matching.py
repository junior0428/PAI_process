import cv2

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QSpinBox,
    QSlider,
    QFormLayout,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


# =======================================================
#               PESTAÑA D3 (Matching)
# =======================================================
class TabD3(QWidget):
    def __init__(self):
        super().__init__()

        # ======== Variables ========
        self.imageA = None
        self.imageB = None
        self.kpA, self.desA = None, None
        self.kpB, self.desB = None, None

        # ======== Panel Izquierdo (Controles) ========
        self.combo_match = QComboBox()
        self.combo_match.addItems(["BF", "FLANN"])

        # Parámetros para Matching
        #   - ratio test, nfeatures, etc.
        self.spin_ratio = QSlider(Qt.Orientation.Horizontal)
        self.spin_ratio.setRange(50, 99)  # 0.5 -> 0.99
        self.spin_ratio.setValue(70)

        self.spin_show = QSpinBox()
        self.spin_show.setRange(1, 500)
        self.spin_show.setValue(50)

        btn_loadA = QPushButton("Cargar Imagen A")
        btn_loadB = QPushButton("Cargar Imagen B")
        btn_match = QPushButton("Hacer Matching")

        btn_loadA.clicked.connect(self.load_imageA)
        btn_loadB.clicked.connect(self.load_imageB)
        btn_match.clicked.connect(self.do_matching)

        form_layout = QFormLayout()
        form_layout.addRow("Método de Matching:", self.combo_match)
        form_layout.addRow(
            "Ratio Test x0.01:", self.spin_ratio
        )  # interpretaremos spin_ratio como ratio/100
        form_layout.addRow("Matches a mostrar:", self.spin_show)

        left_layout = QVBoxLayout()
        left_layout.addLayout(form_layout)
        left_layout.addWidget(btn_loadA)
        left_layout.addWidget(btn_loadB)
        left_layout.addWidget(btn_match)
        left_layout.addStretch()

        # ======== Panel Derecho (Imágenes) ========
        #  - 2 labels arriba para la imagen A y B
        #  - 1 label abajo para el matching result
        self.labelA = QLabel("Imagen A")
        self.labelA.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelA.setStyleSheet("border: 1px solid gray;")
        self.labelA.setMinimumSize(500, 300)

        self.labelB = QLabel("Imagen B")
        self.labelB.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelB.setStyleSheet("border: 1px solid gray;")
        self.labelB.setMinimumSize(500, 300)

        self.labelResult = QLabel("Resultado")
        self.labelResult.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelResult.setStyleSheet("border: 1px solid gray;")
        self.labelResult.setMinimumSize(600, 400)

        right_top_layout = QHBoxLayout()
        right_top_layout.addWidget(self.labelA)
        right_top_layout.addWidget(self.labelB)

        right_layout = QVBoxLayout()
        right_layout.addLayout(right_top_layout)
        right_layout.addWidget(self.labelResult)
        right_layout.addStretch()

        # ======== Layout Principal (Horizontal) ========
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.setLayout(main_layout)

    def load_imageA(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagen A", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.imageA = cv2.imread(file_path)
            if self.imageA is None:
                QMessageBox.critical(self, "Error", "No se pudo cargar la imagen A.")
                return
            self.show_image(self.imageA, self.labelA)
            self.kpA, self.desA = self.extract_sift(self.imageA)
        else:
            QMessageBox.warning(self, "Atención", "No se seleccionó ninguna imagen A.")

    def load_imageB(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagen B", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.imageB = cv2.imread(file_path)
            if self.imageB is None:
                QMessageBox.critical(self, "Error", "No se pudo cargar la imagen B.")
                return
            self.show_image(self.imageB, self.labelB)
            self.kpB, self.desB = self.extract_sift(self.imageB)
        else:
            QMessageBox.warning(self, "Atención", "No se seleccionó ninguna imagen B.")

    def do_matching(self):
        if self.desA is None or self.desB is None:
            QMessageBox.warning(self, "Atención", "Primero carga ambas imágenes.")
            return

        method = self.combo_match.currentText()
        ratio_thr = self.spin_ratio.value() / 100.0
        max_show = self.spin_show.value()

        # BF or FLANN
        # SIFT -> NORM_L2
        bf = None
        if method == "BF":
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(self.desA, self.desB, k=2)
        else:
            # FLANN
            index_params = dict(algorithm=1, trees=5)  # KDTree
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(self.desA, self.desB, k=2)

        good = []
        for m, n in matches:
            if m.distance < ratio_thr * n.distance:
                good.append(m)

        good = sorted(good, key=lambda x: x.distance)
        out = cv2.drawMatches(
            self.imageA,
            self.kpA,
            self.imageB,
            self.kpB,
            good[:max_show],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        self.show_image(out, self.labelResult)

    def extract_sift(self, image):
        sift = cv2.SIFT_create(nfeatures=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des

    def show_image(self, cv_img, label):
        if cv_img is None:
            return
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(
            label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio
        )
        label.setPixmap(scaled)
