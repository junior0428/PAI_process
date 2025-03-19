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
    QSpinBox,
    QFormLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


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
        self.spin_minLine.setValue(30)

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
        self.labelA.setMinimumSize(500, 300)

        self.labelB = QLabel("Imagen B (líneas)")
        self.labelB.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelB.setStyleSheet("border: 1px solid gray;")
        self.labelB.setMinimumSize(500, 300)

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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagen A", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagen B", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
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
            for x1, y1, x2, y2 in self.linesA:
                cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.show_image(temp, self.labelA)

    def detect_linesB(self):
        if self.imageB is None:
            QMessageBox.warning(self, "Atención", "Primero carga la Imagen B.")
            return
        self.linesB = self.detect_lines(self.imageB)
        temp = self.imageB.copy()
        if self.linesB is not None:
            for x1, y1, x2, y2 in self.linesB:
                cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.show_image(temp, self.labelB)

    def detect_lines(self, image):
        c1 = self.spin_canny1.value()
        c2 = self.spin_canny2.value()
        minL = self.spin_minLine.value()
        maxG = self.spin_maxGap.value()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, c1, c2)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 80, minLineLength=minL, maxLineGap=maxG
        )
        if lines is not None:
            lines = lines.reshape(-1, 4)
        return lines

    def match_lines(self):
        if self.linesA is None or self.linesB is None:
            QMessageBox.warning(
                self, "Atención", "Primero detecta líneas en ambas imágenes."
            )
            return

        # Matching muy básico: (rho,theta)
        linesA_rt = [self.line_to_rho_theta(l) for l in self.linesA]
        linesB_rt = [self.line_to_rho_theta(l) for l in self.linesB]

        matched_pairs = []
        for i, (rhoA, thetaA) in enumerate(linesA_rt):
            best_j = -1
            best_dist = 999999
            for j, (rhoB, thetaB) in enumerate(linesB_rt):
                dist = abs(rhoA - rhoB) + abs(thetaA - thetaB) * 10
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
        result[:hB, wA : wA + wB] = self.imageB

        # Dibujar
        for iA, iB in matched_pairs:
            x1A, y1A, x2A, y2A = self.linesA[iA]
            x1B, y1B, x2B, y2B = self.linesB[iB]
            # lineas
            cv2.line(result, (x1A, y1A), (x2A, y2A), (0, 255, 0), 2)
            cv2.line(result, (x1B + wA, y1B), (x2B + wA, y2B), (0, 255, 0), 2)
            # conectar centros
            cxA, cyA = (x1A + x2A) // 2, (y1A + y2A) // 2
            cxB, cyB = (x1B + x2B) // 2 + wA, (y1B + y2B) // 2
            cv2.line(result, (cxA, cyA), (cxB, cyB), (0, 0, 255), 1)

        self.show_image(result, self.labelResult)

    def line_to_rho_theta(self, line):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        theta = np.arctan2(dy, dx)
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
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
        scaled = pixmap.scaled(
            label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio
        )
        label.setPixmap(scaled)
