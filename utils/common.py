# utils.py
import cv2
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import Qt

def load_image_from_dialog(parent, label):
    """
    Abre un diálogo para cargar una imagen y la muestra en el label.
    Devuelve la imagen cargada (o None si se cancela).
    """
    file_path, _ = QFileDialog.getOpenFileName(
        parent, "Abrir Imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
    )
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.critical(parent, "Error", "No se pudo cargar la imagen.")
            return None
        parent.show_image(img, label)
        return img
    else:
        QMessageBox.warning(parent, "Atención", "No se seleccionó ninguna imagen.")
        return None
