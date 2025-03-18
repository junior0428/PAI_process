import cv2
import numpy as np

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
    QFormLayout,
    QDoubleSpinBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


# =======================================================
#               PESTAÑA D2 (Extracción)
# =======================================================
class TabD2(QWidget):
    def __init__(self):
        super().__init__()

        # ======== Variables ========
        self.image = None
        self.kp = None
        self.des = None

        # ======== Panel Izquierdo (Controles) ========
        self.combo_method = QComboBox()
        self.combo_method.addItems(["SIFT", "ORB", "FAST", "Harris"])

        # Parámetros SIFT
        self.spin_sift_nfeatures = QSpinBox()
        self.spin_sift_nfeatures.setRange(1, 20000)
        self.spin_sift_nfeatures.setValue(5000)

        self.spin_sift_octaves = QSpinBox()
        self.spin_sift_octaves.setRange(1, 10)
        self.spin_sift_octaves.setValue(3)

        self.double_sift_contrast = QDoubleSpinBox()
        self.double_sift_contrast.setRange(0.0, 1.0)
        self.double_sift_contrast.setDecimals(4)
        self.double_sift_contrast.setValue(0.04)

        self.double_sift_edge = QDoubleSpinBox()
        self.double_sift_edge.setRange(1.0, 100.0)
        self.double_sift_edge.setDecimals(2)
        self.double_sift_edge.setValue(10.0)

        self.double_sift_sigma = QDoubleSpinBox()
        self.double_sift_sigma.setRange(0.1, 10.0)
        self.double_sift_sigma.setDecimals(2)
        self.double_sift_sigma.setValue(1.6)

        # Parámetro "threshold" para FAST o Harris
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(1, 100)
        self.spin_threshold.setValue(25)

        btn_load = QPushButton("Cargar Imagen")
        btn_extract = QPushButton("Extraer")

        btn_load.clicked.connect(self.load_image)
        btn_extract.clicked.connect(self.extract_features)

        form_layout = QFormLayout()
        form_layout.addRow("Método:", self.combo_method)
        form_layout.addRow("SIFT nFeatures:", self.spin_sift_nfeatures)
        form_layout.addRow("SIFT OctaveLayers:", self.spin_sift_octaves)
        form_layout.addRow("SIFT ContrastThr:", self.double_sift_contrast)
        form_layout.addRow("SIFT EdgeThr:", self.double_sift_edge)
        form_layout.addRow("SIFT Sigma:", self.double_sift_sigma)
        form_layout.addRow("Threshold (FAST/Harris):", self.spin_threshold)

        left_layout = QVBoxLayout()
        left_layout.addLayout(form_layout)
        left_layout.addWidget(btn_load)
        left_layout.addWidget(btn_extract)
        left_layout.addStretch()

        # ======== Panel Derecho (Imágenes) ========
        self.label_image = QLabel("No hay imagen")
        self.label_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_image.setStyleSheet("border: 1px solid gray;")
        self.label_image.setMinimumSize(800, 600)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.label_image)
        right_layout.addStretch()

        # ======== Layout Principal (Horizontal) ========
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)  # Panel izquierdo
        main_layout.addLayout(right_layout, 3)  # Panel derecho (más ancho)

        self.setLayout(main_layout)

    def extract_features(self):
        if self.image is None:
            QMessageBox.warning(self, "Atención", "Primero carga una imagen.")
            return

        method = self.combo_method.currentText()

        # Leemos los parámetros SIFT
        sift_nfeat = self.spin_sift_nfeatures.value()
        sift_octaves = self.spin_sift_octaves.value()
        sift_contrast = self.double_sift_contrast.value()
        sift_edge = self.double_sift_edge.value()
        sift_sigma = self.double_sift_sigma.value()

        # threshold para FAST/Harris
        thres = self.spin_threshold.value()

        # Convertimos a gris
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if method == "SIFT":
            sift = cv2.SIFT_create(
                nfeatures=sift_nfeat,
                nOctaveLayers=sift_octaves,
                contrastThreshold=sift_contrast,
                edgeThreshold=sift_edge,
                sigma=sift_sigma,
            )
            self.kp, self.des = sift.detectAndCompute(gray, None)

        elif method == "ORB":
            orb = cv2.ORB_create(nfeatures=sift_nfeat)
            self.kp = orb.detect(gray, None)
            self.kp, self.des = orb.compute(gray, self.kp)

        elif method == "FAST":
            # threshold = thres
            fast = cv2.FastFeatureDetector_create(thres)
            self.kp = fast.detect(gray, None)
            self.des = None  # FAST no genera descriptores

        elif method == "Harris":
            # threshold = 0.01 * ...
            # harris "manual"
            dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            # threshold relativo a 'thres'
            # Solo a modo de ejemplo, no es muy exacto
            corner_thr = (thres / 100.0) * dst.max()
            kp_list = []
            h, w = gray.shape
            for y in range(h):
                for x in range(w):
                    if dst[y, x] > corner_thr:
                        kp_list.append(cv2.KeyPoint(float(x), float(y), 3.0))
            self.kp = kp_list
            self.des = None

        # Dibujar keypoints en la imagen
        if self.kp is not None:
            temp = cv2.drawKeypoints(
                self.image,
                self.kp,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            self.show_image(temp, self.label_image)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                QMessageBox.critical(self, "Error", "No se pudo cargar la imagen.")
                return
            self.show_image(self.image, self.label_image)
            self.kp, self.des = None, None
        else:
            QMessageBox.warning(self, "Atención", "No se seleccionó ninguna imagen.")

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
