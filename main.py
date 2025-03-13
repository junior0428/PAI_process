import sys
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox, QTabWidget,
    QSpinBox, QSlider, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
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

        # Parámetros de ejemplo
        #   - nfeatures (para SIFT/ORB)
        #   - threshold (para FAST/Harris)
        self.spin_nfeatures = QSpinBox()
        self.spin_nfeatures.setRange(1, 2000)
        self.spin_nfeatures.setValue(500)

        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(1, 100)
        self.spin_threshold.setValue(25)

        btn_load = QPushButton("Cargar Imagen")
        btn_extract = QPushButton("Extraer")

        btn_load.clicked.connect(self.load_image)
        btn_extract.clicked.connect(self.extract_features)

        form_layout = QFormLayout()
        form_layout.addRow("Método:", self.combo_method)
        form_layout.addRow("nFeatures:", self.spin_nfeatures)
        form_layout.addRow("Threshold:", self.spin_threshold)

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
        main_layout.addLayout(left_layout, 1)   # Panel izquierdo
        main_layout.addLayout(right_layout, 3)  # Panel derecho (más ancho)

        self.setLayout(main_layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                QMessageBox.critical(self, "Error", "No se pudo cargar la imagen.")
                return
            self.show_image(self.image, self.label_image)
            self.kp, self.des = None, None
        else:
            QMessageBox.warning(self, "Atención", "No se seleccionó ninguna imagen.")

    def extract_features(self):
        if self.image is None:
            QMessageBox.warning(self, "Atención", "Primero carga una imagen.")
            return

        method = self.combo_method.currentText()
        nfeat = self.spin_nfeatures.value()
        thres = self.spin_threshold.value()

        # Convertimos a gris
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if method == "SIFT":
            sift = cv2.SIFT_create(nfeatures=nfeat)
            self.kp, self.des = sift.detectAndCompute(gray, None)

        elif method == "ORB":
            orb = cv2.ORB_create(nfeatures=nfeat)
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
            temp = cv2.drawKeypoints(self.image, self.kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.show_image(temp, self.label_image)

    def show_image(self, cv_img, label):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(scaled)

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
        form_layout.addRow("Ratio Test x0.01:", self.spin_ratio)  # interpretaremos spin_ratio como ratio/100
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
        self.labelA.setMinimumSize(600, 400)

        self.labelB = QLabel("Imagen B")
        self.labelB.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelB.setStyleSheet("border: 1px solid gray;")
        self.labelB.setMinimumSize(600, 400)

        self.labelResult = QLabel("Resultado")
        self.labelResult.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelResult.setStyleSheet("border: 1px solid gray;")
        self.labelResult.setMinimumSize(700, 500)

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen A", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
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
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen B", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
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
        out = cv2.drawMatches(self.imageA, self.kpA, self.imageB, self.kpB,
                              good[:max_show], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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
        scaled = pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(scaled)

# =======================================================
#               PESTAÑA D9 (Líneas)
# =======================================================
class TabD9(QWidget):
    def __init__(self):
        super().__init__()

        self.imageA = None
        self.imageB = None
        self.linesA = None
        self.linesB = None

        # ======== Panel Izquierdo (Controles) ========
        # Ejemplo de parámetros: thresholds Canny, minLineLength, maxLineGap
        self.spin_canny1 = QSpinBox()
        self.spin_canny1.setRange(1, 500)
        self.spin_canny1.setValue(100)
        self.spin_canny2 = QSpinBox()
        self.spin_canny2.setRange(1, 500)
        self.spin_canny2.setValue(200)

        self.spin_minLine = QSpinBox()
        self.spin_minLine.setRange(1, 500)
        self.spin_minLine.setValue(50)

        self.spin_maxGap = QSpinBox()
        self.spin_maxGap.setRange(1, 200)
        self.spin_maxGap.setValue(10)

        btn_loadA = QPushButton("Cargar Imagen A")
        btn_loadB = QPushButton("Cargar Imagen B")
        btn_linesA = QPushButton("Detectar Líneas (A)")
        btn_linesB = QPushButton("Detectar Líneas (B)")
        btn_match_lines = QPushButton("Matching Líneas")

        btn_loadA.clicked.connect(self.load_imageA)
        btn_loadB.clicked.connect(self.load_imageB)
        btn_linesA.clicked.connect(self.detect_linesA)
        btn_linesB.clicked.connect(self.detect_linesB)
        btn_match_lines.clicked.connect(self.match_lines)

        form_layout = QFormLayout()
        form_layout.addRow("Canny Threshold1:", self.spin_canny1)
        form_layout.addRow("Canny Threshold2:", self.spin_canny2)
        form_layout.addRow("minLineLength:", self.spin_minLine)
        form_layout.addRow("maxLineGap:", self.spin_maxGap)

        left_layout = QVBoxLayout()
        left_layout.addLayout(form_layout)
        left_layout.addWidget(btn_loadA)
        left_layout.addWidget(btn_loadB)
        left_layout.addWidget(btn_linesA)
        left_layout.addWidget(btn_linesB)
        left_layout.addWidget(btn_match_lines)
        left_layout.addStretch()

        # ======== Panel Derecho (Imágenes) ========
        self.labelA = QLabel("Imagen A (líneas)")
        self.labelA.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelA.setStyleSheet("border: 1px solid gray;")
        self.labelA.setMinimumSize(600, 400)

        self.labelB = QLabel("Imagen B (líneas)")
        self.labelB.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelB.setStyleSheet("border: 1px solid gray;")
        self.labelB.setMinimumSize(600, 400)

        self.labelResult = QLabel("Resultado Matching Líneas")
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
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen A", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.imageA = cv2.imread(file_path)
            if self.imageA is None:
                QMessageBox.critical(self, "Error", "No se pudo cargar la imagen A.")
                return
            self.show_image(self.imageA, self.labelA)
            self.linesA = None
        else:
            QMessageBox.warning(self, "Atención", "No se seleccionó ninguna imagen A.")

    def load_imageB(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen B", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.imageB = cv2.imread(file_path)
            if self.imageB is None:
                QMessageBox.critical(self, "Error", "No se pudo cargar la imagen B.")
                return
            self.show_image(self.imageB, self.labelB)
            self.linesB = None
        else:
            QMessageBox.warning(self, "Atención", "No se seleccionó ninguna imagen B.")

    def detect_linesA(self):
        if self.imageA is None:
            QMessageBox.warning(self, "Atención", "Primero carga la Imagen A.")
            return
        self.linesA = self.detect_lines(self.imageA)
        temp = self.imageA.copy()
        if self.linesA is not None:
            for x1,y1,x2,y2 in self.linesA:
                cv2.line(temp, (x1,y1), (x2,y2), (0,255,0), 2)
        self.show_image(temp, self.labelA)

    def detect_linesB(self):
        if self.imageB is None:
            QMessageBox.warning(self, "Atención", "Primero carga la Imagen B.")
            return
        self.linesB = self.detect_lines(self.imageB)
        temp = self.imageB.copy()
        if self.linesB is not None:
            for x1,y1,x2,y2 in self.linesB:
                cv2.line(temp, (x1,y1), (x2,y2), (0,255,0), 2)
        self.show_image(temp, self.labelB)

    def detect_lines(self, image):
        c1 = self.spin_canny1.value()
        c2 = self.spin_canny2.value()
        minL = self.spin_minLine.value()
        maxG = self.spin_maxGap.value()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, c1, c2)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=minL, maxLineGap=maxG)
        if lines is not None:
            lines = lines.reshape(-1,4)
        return lines

    def match_lines(self):
        if self.linesA is None or self.linesB is None:
            QMessageBox.warning(self, "Atención", "Primero detecta líneas en ambas imágenes.")
            return

        # Matching muy básico: (rho,theta)
        linesA_rt = [self.line_to_rho_theta(l) for l in self.linesA]
        linesB_rt = [self.line_to_rho_theta(l) for l in self.linesB]

        matched_pairs = []
        for i, (rhoA, thetaA) in enumerate(linesA_rt):
            best_j = -1
            best_dist = 999999
            for j, (rhoB, thetaB) in enumerate(linesB_rt):
                dist = abs(rhoA - rhoB) + abs(thetaA - thetaB)*10
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j != -1 and best_dist < 50:
                matched_pairs.append((i, best_j))

        # Unir imágenes
        hA, wA = self.imageA.shape[:2]
        hB, wB = self.imageB.shape[:2]
        out_h = max(hA, hB)
        out_w = wA + wB
        result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        result[:hA, :wA] = self.imageA
        result[:hB, wA:wA+wB] = self.imageB

        # Dibujar
        for (iA, iB) in matched_pairs:
            x1A,y1A,x2A,y2A = self.linesA[iA]
            x1B,y1B,x2B,y2B = self.linesB[iB]
            # lineas
            cv2.line(result, (x1A,y1A), (x2A,y2A), (0,255,0), 2)
            cv2.line(result, (x1B+wA,y1B), (x2B+wA,y2B), (0,255,0), 2)
            # conectar centros
            cxA, cyA = (x1A+x2A)//2, (y1A+y2A)//2
            cxB, cyB = (x1B+x2B)//2 + wA, (y1B+y2B)//2
            cv2.line(result, (cxA,cyA), (cxB,cyB), (0,0,255), 1)

        self.show_image(result, self.labelResult)

    def line_to_rho_theta(self, line):
        x1,y1,x2,y2 = line
        dx = x2 - x1
        dy = y2 - y1
        theta = np.arctan2(dy, dx)
        rho = x1*np.cos(theta) + y1*np.sin(theta)
        if rho < 0:
            rho = -rho
            theta += np.pi
        return rho, theta

    def show_image(self, cv_img, label):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(scaled)

# =======================================================
#               VENTANA PRINCIPAL
# =======================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento por Pestañas (con Panel Izquierdo)")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.tab_d2 = TabD2()
        self.tab_d3 = TabD3()
        self.tab_d9 = TabD9()

        self.tabs.addTab(self.tab_d2, "D2: Extracción")
        self.tabs.addTab(self.tab_d3, "D3: Matching")
        self.tabs.addTab(self.tab_d9, "D9: Líneas")

        self.setCentralWidget(self.tabs)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
