from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QCheckBox, QFileDialog, QApplication, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
import cv2
import os
import sys
from ImageProcessing.image_utils import convert_cv_qt

class ImageBrowserWindow(QWidget):
    def __init__(self, parent=None, pipeline=None):
        super().__init__(parent)
        self.setWindowTitle("Image Browser")
        self.resize(800, 600)
        self.pipeline = pipeline 
        
        # Buttons and toggles
        self.left_button = QPushButton("← Previous")
        self.right_button = QPushButton("Next →")
        self.select_folder_button = QPushButton("Select Folder")
        self.toggle_segm = QCheckBox("Predict Segm + CNN")
        self.toggle_yolo = QCheckBox("Predict YOLO")

        self.toggle_segm.setChecked(True)
        self.toggle_yolo.setChecked(False)

        # Image display label
        self.image_label = QLabel("No image loaded")
        self.image_label.setFixedSize(640, 640)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)

        # Layouts
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.left_button)
        controls_layout.addWidget(self.right_button)
        controls_layout.addWidget(self.select_folder_button)
        controls_layout.addWidget(self.toggle_segm)
        controls_layout.addWidget(self.toggle_yolo)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.status_label)

        # Connect signals
        self.left_button.clicked.connect(self.previous_image)
        self.right_button.clicked.connect(self.next_image)
        self.select_folder_button.clicked.connect(self.select_folder_dialog)
        self.toggle_segm.stateChanged.connect(self.show_image)
        self.toggle_yolo.stateChanged.connect(self.show_image)

        self.images = []
        self.current_index = 0

        # Initially disable navigation buttons until folder loaded
        self.left_button.setEnabled(False)
        self.right_button.setEnabled(False)

    @Slot()
    def select_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", os.getcwd())
        if folder:
            self.load_images(folder)

    def load_images(self, folder_path):
        self.images = []
        try:
            files = sorted(
                f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            # Load last 50 images if more
            self.images = [os.path.join(folder_path, f) for f in files[-50:]]
            self.current_index = 0
            if self.images:
                self.status_label.setText(f"Loaded {len(self.images)} images from {folder_path}")
                self.left_button.setEnabled(False)
                self.right_button.setEnabled(len(self.images) > 1)
                self.show_image()
            else:
                self.image_label.setText("No images found in folder")
                self.status_label.setText("")
                self.left_button.setEnabled(False)
                self.right_button.setEnabled(False)
        except Exception as e:
            self.image_label.setText(f"Error: {e}")
            self.status_label.setText("")
            self.left_button.setEnabled(False)
            self.right_button.setEnabled(False)

    def show_image(self):
        if not self.images or self.current_index < 0 or self.current_index >= len(self.images):
            self.image_label.setText("No images loaded")
            return

        img_path = self.images[self.current_index]
        frame = cv2.imread(img_path)
        if frame is None:
            self.image_label.setText(f"Failed loading {img_path}")
            return

        overlayed_img = frame.copy()
        # Check if pipeline is available and either toggle is enabled
        if self.pipeline and (self.toggle_segm.isChecked() or self.toggle_yolo.isChecked()):
            results = self.pipeline.process_image(
                frame,
                predict_segm=self.toggle_segm.isChecked(),
                predict_class=True,
                predict_yolo=self.toggle_yolo.isChecked()
            )
            if results.get('segmentation_mask') is not None:
                try:
                    overlayed_img = self.pipeline.segm_model.create_segmentation_overlay(overlayed_img, results['segmentation_mask'])
                except Exception:
                    pass
            if results.get('yolo_detections') is not None:
                try:
                    overlayed_img = self.pipeline.yolo_model.create_detection_overlay(overlayed_img, results['yolo_detections'])
                except Exception:
                    pass
        else:
            overlayed_img = frame

        pix = convert_cv_qt(overlayed_img, self.image_label.width(), self.image_label.height())
        self.image_label.setPixmap(pix)
        self.status_label.setText(f"Showing image {self.current_index + 1} of {len(self.images)}: {os.path.basename(img_path)}")

        self.left_button.setEnabled(self.current_index > 0)
        self.right_button.setEnabled(self.current_index < len(self.images) - 1)

    @Slot()
    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    @Slot()
    def next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.show_image()

    def apply_prediction_overlay(self, img, img_path):
        """Dummy overlay function: draws colored rectangles and text if toggles are active."""
        overlay_img = img.copy()
        h, w = overlay_img.shape[:2]

        painter = QPainter()
        # Convert to QImage for painting
        temp_img = QImage(overlay_img.data, w, h, overlay_img.strides[0], QImage.Format.Format_BGR888)
        painter.begin(temp_img)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)

        if self.toggle_segm.isChecked():
            painter.setPen(QColor(0, 255, 0))
            painter.drawRect(int(w * 0.1), int(h * 0.1), int(w * 0.3), int(h * 0.3))
            painter.drawText(int(w * 0.12), int(h * 0.1) + 20, "Segm + CNN")

        if self.toggle_yolo.isChecked():
            painter.setPen(QColor(255, 0, 0))
            painter.drawRect(int(w * 0.5), int(h * 0.4), int(w * 0.3), int(h * 0.3))
            painter.drawText(int(w * 0.52), int(h * 0.4) + 20, "YOLO")

        painter.end()

        # Convert QImage back to OpenCV BGR numpy array
        ptr = temp_img.bits()
        ptr.setsize(temp_img.byteCount())
        overlay_img = np.array(ptr).reshape((temp_img.height(), temp_img.width(), 4))
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGR)

        return overlay_img
