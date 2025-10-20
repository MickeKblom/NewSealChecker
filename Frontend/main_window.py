from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QApplication, QCheckBox, QDialog, QSlider
)
from PySide6.QtCore import Slot, Qt, QRect, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
import numpy as np
from ImageProcessing.inference import ImageInferencePipeline
from ImageProcessing.image_utils import convert_cv_qt, generate_unique_image_id, save_yolo_predictions, save_segmentation_predictions, save_cnn_cropped, save_cnn_predictions
from Interfaces.ODPredict import ODClassifierWrapper
from Frontend.settings_dialog import SettingsDialog
#from Acquisition.camera_worker_webcam import WebcamWorker
#from Acquisition.camera_worker import CameraWorker
from Acquisition.camera_worker_tiscam import CameraWorker
from ImageProcessing.processing_worker import ProcessingWorker
from Frontend.image_browser import ImageBrowserWindow
from ImageProcessing.image_utils import generate_class_colors
from PIL import Image
from datetime import datetime
import time
from types_shared import ProcessFlags, FramePacket


class MainWindow(QWidget):
    # UI → Worker signals must be declared at class level
    snapRequested = Signal()
    liveStartRequested = Signal()
    liveStopRequested = Signal()
    exposureRequested = Signal(int)

    processRequested = Signal(object, object)
    segmParamsUpdated = Signal(dict)
    cnnParamsUpdated = Signal(dict)

    def __init__(self):
        super().__init__()
        
       
        # Segmentation and YOLO params dicts - initialize defaults (used by Settings)
        self.segm_params = {
            "confidence_threshold": 0.3,
            "temperature": 1.5,
            "min_area": 100,
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
        
        self.num_classes = 1  # or dynamically set based on model
        self.class_colors = generate_class_colors(self.num_classes)


        self.setWindowTitle("Camera Application")


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

        self.toggle_predict_cnn = QCheckBox("Enable CNN Classification")
        self.toggle_predict_cnn.setChecked(False)

        self.toggle_predict_yolo = QCheckBox("Enable YOLO Prediction")
        self.toggle_predict_yolo.setChecked(False)

        # Debug: show CNN inputs (cropped+resized image and mask) on the right
        self.debug_cnn_checkbox = QCheckBox("Show CNN inputs")
        self.debug_cnn_checkbox.setChecked(False)

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

        # Layout: image area with optional right debug panel
        main_layout = QVBoxLayout(self)

        image_row = QHBoxLayout()

        # Right-side debug panel (hidden by default)
        self.cnn_debug_panel = QVBoxLayout()
        self.cnn_img_label = QLabel("CNN img")
        self.cnn_img_label.setFixedSize(256, 112)  # half of 512x224
        self.cnn_img_label.setAlignment(Qt.AlignCenter)
        self.cnn_mask_label = QLabel("CNN mask")
        self.cnn_mask_label.setFixedSize(256, 112)
        self.cnn_mask_label.setAlignment(Qt.AlignCenter)
        right_panel_widget = QVBoxLayout()
        right_panel_widget.addWidget(self.cnn_img_label)
        right_panel_widget.addWidget(self.cnn_mask_label)

        # Container widget to be able to hide/show easily
        from PySide6.QtWidgets import QWidget as _QW
        self.right_panel_container = _QW()
        self.right_panel_container.setLayout(right_panel_widget)
        self.right_panel_container.setVisible(False)

        image_row.addWidget(self.image_label, stretch=3)
        image_row.addWidget(self.right_panel_container, stretch=2)
        main_layout.addLayout(image_row)

        row1 = QHBoxLayout()
        row1.addWidget(self.live_mode_button)
        row1.addWidget(self.capture_button)

        row2 = QHBoxLayout()
        row2.addWidget(self.toggle_predict_segm)
        row2.addWidget(self.toggle_predict_cnn)
        row2.addWidget(self.toggle_predict_yolo)
        row2.addWidget(self.open_browser_button)
        row2.addWidget(self.settings_button)
        row2.addWidget(self.debug_cnn_checkbox)
       


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

        # ---------- UI connections ----------
        self.capture_batch_button.clicked.connect(self.capture_batch_image)
        self.process_batch_button.clicked.connect(self.process_batch_images)
        self.reset_batch_button.clicked.connect(self.reset_batch)

        self.open_browser_button.clicked.connect(self.open_browser)
        # Ensure cnn_params exists before connecting debug toggle
        if not hasattr(self, 'cnn_params') or not isinstance(getattr(self, 'cnn_params'), dict):
            self.cnn_params = {"cnn_threshold": 0.5, "debug_show_inputs": False}
        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.debug_cnn_checkbox.toggled.connect(self.on_debug_cnn_toggled)

        # Emit our signals from handlers (UI → Worker)
        self.capture_button.clicked.connect(self.on_snap_clicked)
        self.live_mode_button.toggled.connect(self.on_live_toggled)
        self.exposure_slider.valueChanged.connect(self.on_exposure_changed)

        # ---------- Threads & Worker ----------
        self.cam_thread = QThread(self)
        #self.camera = WebcamWorker(width=640, height=640, exposure_us=29000, timeout_ms=1000)
        self.camera = CameraWorker(width=640, height=640, exposure_us=29000, timeout_ms=1000)
        self.camera.moveToThread(self.cam_thread)
        self.proc_thread = QThread(self)
        self.processor = ProcessingWorker(
            segm_model_path="Models/Segmentation/best.pth",
            cnn_model_path="Models/CNN/bestCheckpoint2.pth",
            yolo_model_path="Models/YOLO/best.pt",
            warmup=True
        )
        self.processor.moveToThread(self.proc_thread)

        # Connect BEFORE starting the thread
        self.cam_thread.started.connect(self.camera.initialize, type=Qt.QueuedConnection)
        self.proc_thread.started.connect(self.processor.initialize, type=Qt.QueuedConnection)

        # UI → Worker (queued)
        self.snapRequested.connect(self.camera.capture_frame, type=Qt.QueuedConnection)
        self.exposureRequested.connect(self.camera.set_exposure, type=Qt.QueuedConnection)

        # UI → Processor (queued)
        self.processRequested.connect(self.processor.process, type=Qt.QueuedConnection)

        # Worker → UI
        self.camera.frame_captured.connect(self.on_frame_captured)
        self.camera.error.connect(self.on_error)
        self.camera.ready.connect(lambda: print("Camera ready"))

        # Processor → UI
        self.processor.processed.connect(self.on_processed)
        self.processor.error.connect(self.on_error)
        self.processor.ready.connect(lambda: print("Processor ready"))

        # UI → Processor params (queued)
        self.segmParamsUpdated.connect(self.processor.update_segm_params, type=Qt.QueuedConnection)
        self.cnnParamsUpdated.connect(self.processor.update_cnn_params, type=Qt.QueuedConnection)


        # Start the threads
        self.cam_thread.start()
        self.proc_thread.start()

        # ---------- State ----------
        self.live_mode = False
        self.last_captured_image = None
        self.image_browser = None
        self._snap_in_progress = False
        
        self.live_enabled = False
        self._frame_inflight = False         # ensures at most one capture is in flight
        self.live_cap_fps = 1.0              # optional cap; e.g., 1 FPS. Set to None/0 to disable


    # ---------- UI Handlers ----------
    @Slot()
    def on_snap_clicked(self):
        if not self._snap_in_progress:
            self._snap_in_progress = True
            self.snapRequested.emit()

    @Slot(bool)
    def on_live_toggled(self, enabled: bool):
        self.live_enabled = enabled

        if enabled:
            # Kick off the first request immediately (if none in flight)
            self._request_next_live_frame()
        else:
            # Stop requesting further frames
            self._frame_inflight = False  # gate; prevents any pending emission
            # (If a capture is already in flight, it will finish; we just won't re-arm.)
            

    def _request_next_live_frame(self):
        if not self.live_enabled or self._frame_inflight:
            return
        self._frame_inflight = True
        self.snapRequested.emit()  # capture_single() in camera thread

    @Slot(int)
    def on_exposure_changed(self, value: int):
        self.exposure_value_label.setText(str(value))
        # Send to camera thread
        self.exposureRequested.emit(int(value))

    # ---------- Camera → UI ----------
    @Slot(object)
    def on_frame_captured(self, packet: FramePacket):
        """UI thread: forward to processing worker with current flags."""
        want_cnn = self.toggle_predict_cnn.isChecked()
        want_segm = self.toggle_predict_segm.isChecked() or want_cnn  # CNN requires a mask
        flags = ProcessFlags(
            predict_segm=want_segm,
            predict_class=want_cnn,
            predict_yolo=self.toggle_predict_yolo.isChecked(),
            live=self.live_mode_button.isChecked(),
        )
        self.processRequested.emit(packet, flags)


    @Slot(str)
    def on_error(self, msg: str):
        print("[ERROR]", msg)

    # ---------- Optional: close cleanly ----------
    def closeEvent(self, event):
        try:
            self.liveStopRequested.emit()
            # Invoke worker shutdowns if available
            try:
                self.camera.shutdown()
            except Exception:
                pass
            try:
                self.processor.shutdown()
            except Exception:
                pass
        except Exception:
            pass
        self.cam_thread.quit()
        self.cam_thread.wait(2000)
        self.proc_thread.quit()
        self.proc_thread.wait(2000)
        super().closeEvent(event)

# ---------- Processor → UI ----------
    @Slot(object)
    def on_processed(self, out):
        """
        out: ProcessedResult
        Update UI, show overlay, optionally save predictions, re-enable UI.
        """
        # Update your image and timing overlays
        acq_s = out.acq_ms / 1000.0
        proc_s = out.proc_ms / 1000.0
        # Map class prediction (0->OK, 1->SHORT) to display text
        pred_cls = out.results.get('class_prediction', None)
        if pred_cls is not None:
            self.current_predicted_class = "SHORT" if int(pred_cls) == 1 else "OK"
        else:
            self.current_predicted_class = ""

        self.update_image(out.overlay_bgr, out.acq_ms, out.proc_ms)

        # Update CNN debug previews if enabled and provided
        if self.debug_cnn_checkbox.isChecked():
            img_np = out.results.get('cnn_input_img_np')
            mask_np = out.results.get('cnn_input_mask_np')
            if img_np is not None:
                pm = convert_cv_qt(img_np, self.cnn_img_label.width(), self.cnn_img_label.height())
                self.cnn_img_label.setPixmap(pm)
            else:
                self.cnn_img_label.clear()
                self.cnn_img_label.setText("CNN img")
            if mask_np is not None:
                # Convert [H,W] -> [H,W,3] grayscale for display
                if mask_np.ndim == 2:
                    import numpy as _np
                    mask_rgb = _np.stack([mask_np * 255] * 3, axis=2).astype('uint8')
                else:
                    mask_rgb = mask_np
                pm2 = convert_cv_qt(mask_rgb, self.cnn_mask_label.width(), self.cnn_mask_label.height())
                self.cnn_mask_label.setPixmap(pm2)
            else:
                self.cnn_mask_label.clear()
                self.cnn_mask_label.setText("CNN mask")

        # Optional saving as in your old code
        if self.toggle_save_predictions.isChecked():
            base_path = "Images/Saved Images"
            image_id = generate_unique_image_id()  # your function
            frame = out.raw_bgr
            if out.results.get('yolo_detections') is not None:
                save_yolo_predictions(frame, out.results['yolo_detections'], base_path, image_id)
            if out.results.get('segmentation_mask') is not None:
                save_segmentation_predictions(frame, out.results['segmentation_mask'], base_path, image_id)
            if out.results.get('cropped_img') is not None and out.results.get('cropped_mask') is not None:
                save_cnn_cropped(out.results['cropped_img'], out.results['cropped_mask'], base_path, image_id)
            # Save CNN predictions (cropped and resized images/masks) in OK/shorts folders
            if (out.results.get('cnn_input_img') is not None and 
                out.results.get('cnn_input_mask') is not None and 
                out.results.get('class_prediction') is not None):
                save_cnn_predictions(
                    out.results['cnn_input_img'], 
                    out.results['cnn_input_mask'], 
                    out.results['class_prediction'], 
                    base_path, 
                    image_id
                )

        # Re-arm processing-paced live: request a NEW capture (no buffer, no cap)
        self._frame_inflight = False
        if self.live_enabled:
            self._request_next_live_frame()

        # Housekeeping for single snap
        self._snap_in_progress = False
        self.toggle_predict_segm.setEnabled(True)
        self.toggle_predict_cnn.setEnabled(True)
        self.toggle_predict_yolo.setEnabled(True)


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
    def update_image(self, frame: np.ndarray, acquisition_time_ms=None, processing_time_ms=None):
        self.last_captured_image = frame
        qt_img = convert_cv_qt(frame, self.image_label.width(), self.image_label.height())

        # Get current timestamp string
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine predicted class text (example uses stored attribute, adjust as needed)
        predicted_class = getattr(self, "current_predicted_class", 0)  # default 0
        
        # Draw overlay text on the pixmap
        qt_img = self.draw_overlay_text(qt_img, timestamp_str, predicted_class, acquisition_time_ms, processing_time_ms)


        # If guideline toggle is ON, draw the guideline semi-transparent over the image
        if self.toggle_guideline.isChecked():
            blended_pix = self.blend_pixmaps(qt_img, self.guideline_pixmap, opacity=0.3)
            self.image_label.setPixmap(blended_pix)
        else:
            self.image_label.setPixmap(qt_img)

        # Keep right panel visibility synced with checkbox
        self.right_panel_container.setVisible(self.debug_cnn_checkbox.isChecked())


    @Slot()
    def capture_batch_image(self):
        frame = self.camera_app.capture_frame()
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

    
    def draw_overlay_text(
        self,
        pixmap: QPixmap | QImage,
        timestamp: str,
        predicted_class: int | str,
        acquisition_time_ms: int | None,
        processing_time_ms: int | None
    ) -> QPixmap:
        """Draw a single top bar: [Time] [Acq xx ms] [Proc yy ms] ......... [RESULT]"""

        # Ensure we are painting on a QPixmap
        if isinstance(pixmap, QImage):
            pm = QPixmap.fromImage(pixmap)
        else:
            pm = pixmap

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)

        # Font & metrics
        font = QFont("Arial", 14, QFont.Weight.Bold)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        # Colors
        text_color = QColor(255, 255, 255)   # white for normal fields
        ok_color = QColor(0, 220, 0)         # green
        bad_color = QColor(230, 30, 30)      # red

        # Compose left-to-right labels as strings
        time_text = timestamp
        acq_text = f"Acq {acquisition_time_ms} ms" if acquisition_time_ms is not None else "Acq -- ms"
        proc_text = f"Proc {processing_time_ms} ms" if processing_time_ms is not None else "Proc -- ms"

        # Result text & color
        if isinstance(predicted_class, str):
            cls = predicted_class.upper()
            result_text = cls
            result_color = bad_color if result_text == "SHORT" else ok_color
        else:
            result_text = ""
            result_color=text_color

        # Layout parameters
        pad_x = 10
        pad_y = 6
        gap = 20                                 # gap between fields on the left
        bar_h = metrics.height() + pad_y * 2     # tall enough for one line

        # Draw the full-width top bar (semi-transparent black)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRect(QRect(0, 0, pm.width(), bar_h))

        # Baseline for text
        baseline_y = pad_y + metrics.ascent()

        # LEFT cluster: time, acq, proc
        painter.setPen(text_color)
        x = pad_x
        painter.drawText(x, baseline_y, time_text)
        x += metrics.horizontalAdvance(time_text) + gap

        painter.drawText(x, baseline_y, acq_text)
        x += metrics.horizontalAdvance(acq_text) + gap

        painter.drawText(x, baseline_y, proc_text)

        # RIGHT: result, right-aligned with padding
        res_w = metrics.horizontalAdvance(result_text)
        res_x = pm.width() - pad_x - res_w
        painter.setPen(result_color)
        painter.drawText(res_x, baseline_y, result_text)

        painter.end()
        return pm


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


    @Slot(bool)
    def on_debug_cnn_toggled(self, enabled: bool):
        # Update worker param and show/hide panel
        if not hasattr(self, 'cnn_params') or not isinstance(getattr(self, 'cnn_params'), dict):
            self.cnn_params = {"cnn_threshold": 0.5, "debug_show_inputs": False}
        self.cnn_params["debug_show_inputs"] = bool(enabled)
        self.cnnParamsUpdated.emit(dict(self.cnn_params))
        self.right_panel_container.setVisible(bool(enabled))


    def open_settings_dialog(self):
        if not hasattr(self, 'cnn_params') or not isinstance(getattr(self, 'cnn_params'), dict):
            self.cnn_params = {"cnn_threshold": 0.5, "debug_show_inputs": False}
        dlg = SettingsDialog(segm_params=self.segm_params, yolo_params=self.yolo_params, cnn_params=self.cnn_params, parent=self)
        if dlg.exec() == QDialog.Accepted:
            segm_updated, yolo_updated, cnn_updated = dlg.get_params()
            self.segm_params.update(segm_updated)
            self.yolo_params.update(yolo_updated)
            self.cnn_params.update(cnn_updated)
            # Propagate segmentation params to processing pipeline (thread-safe)
            self.segmParamsUpdated.emit(dict(self.segm_params))
            self.cnnParamsUpdated.emit(dict(self.cnn_params))
            # Show/hide right panel based on debug flag
            dbg = bool(self.cnn_params.get("debug_show_inputs", False))
            self.right_panel_container.setVisible(dbg)

    def open_browser(self):
        if self.image_browser is None or not self.image_browser.isVisible():
            # Pass pipeline or other needed references here if required
            self.image_browser = ImageBrowserWindow(pipeline=self.pipeline)
        self.image_browser.show()
        self.image_browser.raise_()
        self.image_browser.activateWindow()
    
    def create_segmentation_overlay(self, frame, mask):
        unique_classes = np.unique(mask)
        print("Unique classes in mask:", unique_classes)
        assert np.all(np.isin(unique_classes, list(self.class_colors.keys()))), "Mask contains invalid class IDs."
        
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in self.class_colors.items():
            overlay[mask == class_id] = color

        # Resize overlay to match frame dimensions
        overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        return cv2.addWeighted(frame, 1 - self.overlay_intensity, overlay, self.overlay_intensity, 0)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
