from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QApplication, QCheckBox, QDialog, QSlider
)
from PySide6.QtCore import Slot, Qt, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
import numpy as np
from ImageProcessing.inference import ImageInferencePipeline
from ImageProcessing.image_utils import convert_cv_qt, generate_unique_image_id, save_yolo_predictions, save_segmentation_predictions, save_cnn_cropped
from Interfaces.segm import SegmentationModel
from Interfaces.ODPredict import ODClassifierWrapper
from Frontend.settings_dialog import SettingsDialog
from Acquisition.camera_interface import CameraApp  # Your camera wrapper module
from Frontend.image_browser import ImageBrowserWindow
from ImageProcessing.image_utils import display_cropped_and_mask
from PIL import Image
from datetime import datetime
import time

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.pipeline = ImageInferencePipeline(
            segm_model_path="Models/Segmentation/best.pth",
            cnn_model_path="Models/CNN/best.pth",
            yolo_model_path="Models/YOLO/best.pt"
        )

                # Segmentation and YOLO params dicts - initialize defaults
        self.segm_params = {
            "confidence_threshold": 0.3,
            "temperature": 1.5,
            "min_area": 100,
            # add other segmentation params here as keys with default float values
        }

        self.yolo_params = {
            "seal": 0.3,
            "od": 0.3,
            "short": 0.3,
            "spring_spot": 0.3,
            "spring": 0.3,
        }


                # Batch processing related state
        self.batch_images = []
        self.batch_counter = 0
        self.max_batch_size = 12  # or any X images

        self.setWindowTitle("Camera Application")

        self.camera_app = CameraApp()
        self.camera_app.new_frame.connect(self.update_image)

        # Image display label
        self.image_label = QLabel("No image")
        self.image_label.setFixedSize(640, 640)
        self.image_label.setAlignment(Qt.AlignCenter)
        # Label to show exposure time title
        self.exposure_label = QLabel("Exposure Time (µs):")
        
        # Load the guideline pixmap once and scale for display
        self.guideline_pixmap = QPixmap(r"D:\NewApplication\Frontend\example_img.jpg")

        self.toggle_guideline = QCheckBox("Show Guideline")
        self.toggle_guideline.stateChanged.connect(self.update_display_with_guideline)

        # Control buttons
        self.live_mode_button = QCheckBox("Start Live Mode")
        self.capture_button = QPushButton("Capture Image")
        self.settings_button = QPushButton("Settings")

        self.toggle_predict_segm = QCheckBox("Enable Segm Prediction")
        self.toggle_predict_segm.setChecked(False)  # Default off

        self.toggle_predict_yolo = QCheckBox("Enable YOLO Prediction")
        self.toggle_predict_yolo.setChecked(False)

        # Create batch control buttons
        self.capture_batch_button = QPushButton("Capture batch image 1")
        self.process_batch_button = QPushButton("Process batch")
        self.reset_batch_button = QPushButton("Reset batch")


        self.open_browser_button = QPushButton("Open Image Browser")
        # Slider creation
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(100)
        self.exposure_slider.setMaximum(150000)
        self.exposure_slider.setSingleStep(200)
        self.exposure_slider.setValue(27000)  # default value
        self.exposure_slider.setTickInterval(10000)
        self.exposure_slider.setTickPosition(QSlider.TicksBelow)
        # Label to show the current value of the slider
        self.exposure_value_label = QLabel(str(self.exposure_slider.value()))

        # Create the toggle button
        self.toggle_save_predictions = QPushButton("Save Predictions", self)
        self.toggle_save_predictions.setCheckable(True)
        self.toggle_save_predictions.setGeometry(10, 10, 150, 30) 

        # Layout (single main layout only)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.image_label)

        row1 = QHBoxLayout()
        row1.addWidget(self.live_mode_button)
        row1.addWidget(self.capture_button)

        row2 = QHBoxLayout()
        row2.addWidget(self.toggle_predict_segm)
        row2.addWidget(self.toggle_predict_yolo)
        row2.addWidget(self.open_browser_button)
        row2.addWidget(self.settings_button)
       


        row3 = QHBoxLayout()
        row3.addWidget(self.exposure_label)
        row3.addWidget(self.exposure_slider)
        row3.addWidget(self.exposure_value_label)
        row3.addWidget(self.toggle_guideline)
        row3.addWidget(self.toggle_save_predictions)


        # Add buttons to a suitable layout (for example in row1 or a new row)
        row4 = QHBoxLayout()
        row4.addWidget(self.capture_batch_button)
        row4.addWidget(self.process_batch_button)
        row4.addWidget(self.reset_batch_button)

        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addLayout(row3)
        main_layout.addLayout(row4)

        # Connect signals
        self.capture_batch_button.clicked.connect(self.capture_batch_image)
        self.process_batch_button.clicked.connect(self.process_batch_images)
        self.reset_batch_button.clicked.connect(self.reset_batch)

        # Connect slider value change to update label and set exposure
        self.exposure_slider.valueChanged.connect(self.update_exposure_time)

        # Connect signals to slots
        self.live_mode_button.clicked.connect(self.toggle_live_mode)
        self.capture_button.clicked.connect(self.capture_image)
        self.open_browser_button.clicked.connect(self.open_browser)
        self.settings_button.clicked.connect(self.open_settings_dialog)

        self.live_mode = False
        self.last_captured_image = None
        self.image_browser = None

    @Slot(int)
    def update_exposure_time(self, value):
        self.exposure_value_label.setText(str(value))
        # Here you call your camera interface method to set exposure time
        self.camera_app.set_exposure(value)  # assuming camera_app has this method

    @Slot()
    def update_display_with_guideline(self):
        # When toggling, refresh image to include or exclude guideline
        if self.last_captured_image is not None:
            self.update_image(self.last_captured_image)

    @Slot(np.ndarray)
    def update_image(self, frame: np.ndarray, acquisition_time=None, processing_time=None):
        self.last_captured_image = frame
        qt_img = convert_cv_qt(frame, self.image_label.width(), self.image_label.height())

        # Get current timestamp string
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine predicted class text (example uses stored attribute, adjust as needed)
        predicted_class = getattr(self, "current_predicted_class", 0)  # default 0
        
        # Draw overlay text on the pixmap
        qt_img = self.draw_overlay_text(qt_img, timestamp_str, predicted_class, acquisition_time, processing_time)


        # If guideline toggle is ON, draw the guideline semi-transparent over the image
        if self.toggle_guideline.isChecked():
            blended_pix = self.blend_pixmaps(qt_img, self.guideline_pixmap, opacity=0.3)
            self.image_label.setPixmap(blended_pix)
        else:
            self.image_label.setPixmap(qt_img)
    @Slot()
    def capture_batch_image(self):
        frame = self.camera_app.capture_single_image()
        if frame is not None:
            if len(self.batch_images) < self.max_batch_size:
                self.batch_images.append(frame)
                self.batch_counter += 1
                self.capture_batch_button.setText(f"Capture batch image {self.batch_counter + 1}")
            else:
                print("Batch is full. Please process or reset before capturing more.")

    @Slot()
    def process_batch_images(self):
        if self.batch_images:
            print(f"Processing batch of {len(self.batch_images)} images (placeholder).")
            # Placeholder: call your batch processing (to be optimized with TensorRT)
            # e.g. results = self.pipeline.process_batch(self.batch_images)
        else:
            print("No images in batch to process.")

    @Slot()
    def reset_batch(self):
        self.batch_images.clear()
        self.batch_counter = 0
        self.capture_batch_button.setText("Capture batch image 1")
        print("Batch reset.")

    def draw_overlay_text(self, pixmap: QPixmap, timestamp: str, predicted_class: int) -> QPixmap:
        # Create a painter to draw on the pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set font and color for the overlay text
        font = QFont("Arial", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))  # White text
        
        # Optional: draw a semi-transparent black background for text readability
        def draw_text_with_background(text, x, y, align_center=False):
            metrics = painter.fontMetrics()
            width = metrics.horizontalAdvance(text)
            height = metrics.height()
            rect_x = x
            if align_center:
                rect_x = x - width // 2
            rect = QRect(rect_x - 5, y - height + 5, width + 10, height + 10)
            painter.fillRect(rect, QColor(0, 0, 0, 150))  # semi-transparent black
            if align_center:
                painter.drawText(rect_x, y, text)
            else:
                painter.drawText(x, y, text)
        
        # Draw timestamp at top-left corner (10 px from edges)
        draw_text_with_background(timestamp, 10, 25)
        
        # Draw predicted class text at top-center
        if predicted_class == 1:
            class_text = "SHORT"
        else:
            class_text = ""
        if class_text:
            center_x = pixmap.width() // 2
            draw_text_with_background(class_text, center_x, 25, align_center=True)
        
        painter.end()
        return pixmap

    def blend_pixmaps(
        self,
        main_pixmap: QPixmap,          # camera frame
        guide_pixmap: QPixmap,          # guideline
        opacity: float = 0.3,
        offset: tuple[int, int] = (0, 0),   # (dx, dy) to move overlay
        center_overlay: bool = True
    ) -> QPixmap:
        # Convert to ARGB premultiplied for correct alpha blending
        main_img = main_pixmap.toImage().convertToFormat(QImage.Format_ARGB32_Premultiplied)

        guide_img = guide_pixmap.toImage().scaled(main_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)


        # Scale guide to fit (keep aspect) — you can change to IgnoreAspectRatio if you want a fill
        guide_img = guide_img.scaled(main_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Compute position
        if center_overlay:
            x = (main_img.width() - guide_img.width()) // 2 + offset[0]
            y = (main_img.height() - guide_img.height()) // 2 + offset[1]
        else:
            x, y = offset

        # Create output
        out = QImage(main_img.size(), QImage.Format_ARGB32_Premultiplied)
        out.fill(Qt.transparent)

        painter = QPainter(out)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 1) Draw camera (opaque background)
        painter.setOpacity(1.0)
        painter.drawImage(0, 0, main_img)

        # 2) Draw guideline (semi-transparent overlay)
        painter.setOpacity(opacity)
        painter.drawImage(x, y, guide_img)

        painter.end()
        return QPixmap.fromImage(out)


    def toggle_live_mode(self):
        if not self.live_mode:
            self.camera_app.start_live_mode()
            self.live_mode_button.setText("Stop Live Mode")
        else:
            self.camera_app.stop_live_mode()
            self.live_mode_button.setText("Start Live Mode")
        self.live_mode = not self.live_mode

    def capture_image(self):
        acquisition_start = time.perf_counter()
        frame = self.camera_app.capture_single_image()
        acquisition_end = time.perf_counter()
        acquisition_time = acquisition_end - acquisition_start

        if frame is not None:
            processing_start = time.perf_counter()
            results = self.pipeline.process_image(
                frame,
                predict_segm=self.toggle_predict_segm.isChecked(),
                predict_class=True,  # or from UI state
                predict_yolo=self.toggle_predict_yolo.isChecked()
            )
            processing_end = time.perf_counter()
            processing_time = processing_end - processing_start

            overlayed_img = frame.copy()
            
            if results['segmentation_mask'] is not None:
                overlayed_img = self.pipeline.segm_model.create_segmentation_overlay(overlayed_img, results['segmentation_mask'])

            if results['yolo_detections'] is not None:
                overlayed_img = self.pipeline.yolo_model.create_detection_overlay(overlayed_img, results['yolo_detections'])

            # Pass times to update_image for overlay drawing
            self.update_image(overlayed_img, acquisition_time, processing_time)

            pix = convert_cv_qt(overlayed_img, self.image_label.width(), self.image_label.height())
            self.image_label.setPixmap(pix)
        


            if self.toggle_save_predictions.isChecked():
                base_path = "Images/Saved Images"
                image_id = generate_unique_image_id()  # You implement this, e.g. timestamp or counter
                
                if results['yolo_detections'] is not None:
                    save_yolo_predictions(frame, results['yolo_detections'], base_path, image_id)
                if results['segmentation_mask'] is not None:
                    save_segmentation_predictions(frame, results['segmentation_mask'], base_path, image_id)
                if results.get('cropped_img') is not None and results.get('cropped_mask') is not None:
                    save_cnn_cropped(results['cropped_img'], results['cropped_mask'], base_path, image_id)
           
            # Enable or disable buttons as needed
            self.toggle_predict_segm.setEnabled(True)
            self.toggle_predict_yolo.setEnabled(True)


    def open_settings_dialog(self):
        dlg = SettingsDialog(segm_params=self.segm_params, yolo_params=self.yolo_params, parent=self)
        if dlg.exec() == QDialog.Accepted:
            new_params = dlg.get_params()
            self.segm_params.update(new_params)
            self.yolo_params.update(new_params)

    def open_browser(self):
        if self.image_browser is None or not self.image_browser.isVisible():
            # Pass pipeline or other needed references here if required
            self.image_browser = ImageBrowserWindow(pipeline=self.pipeline)
        self.image_browser.show()
        self.image_browser.raise_()
        self.image_browser.activateWindow()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
