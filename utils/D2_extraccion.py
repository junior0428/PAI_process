from PyQt6.QtWidgets import QFileDialog, QMessageBox
import cv2


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
