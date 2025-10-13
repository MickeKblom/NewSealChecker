from PySide6.QtWidgets import QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QHBoxLayout

class SettingsDialog(QDialog):
    def __init__(self, segm_model, parent=None):
        super().__init__(parent)
        self.segm_model = segm_model
        self.setWindowTitle("Segmentation Model Settings")

        layout = QFormLayout(self)

        # Confidence threshold
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setValue(self.segm_model.confidence_threshold)
        layout.addRow("Confidence Threshold:", self.confidence_spin)

        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 5.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(self.segm_model.temperature)
        layout.addRow("Temperature:", self.temperature_spin)

        # Minimum area
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 10000)
        self.min_area_spin.setValue(self.segm_model.min_area)
        layout.addRow("Min Area:", self.min_area_spin)


        # Dilation
        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(1, 50)
        self.dilation_spin.setValue(self.segm_model.dilation)
        layout.addRow("Dilation:", self.dilation)

        # overlay_intensity (overlay transparency)
        self.overlay_intensity_spin = QDoubleSpinBox()
        self.overlay_intensity_spin.setRange(0.0, 1.0)
        self.overlay_intensity_spin.setSingleStep(0.05)
        self.overlay_intensity_spin.setValue(self.segm_model.overlay_intensity)
        layout.addRow("Overlay Transparency:", self.overlay_intensity_spin)

        # Enable shape filtering
        self.shape_filter_checkbox = QCheckBox()
        self.shape_filter_checkbox.setChecked(self.segm_model.enable_shape_filtering)
        layout.addRow("Enable Shape Filtering:", self.shape_filter_checkbox)

        # Buttons for OK and Cancel
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addRow(button_layout)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def accept(self):
        # Update the segmentation model parameters
        self.segm_model.set_confidence_threshold(self.confidence_spin.value())
        self.segm_model.set_temperature(self.temperature_spin.value())
        self.segm_model.set_min_area(self.min_area_spin.value())
        self.segm_model.set_overlay_intensity(self.overlay_intensity_spin.value())
        self.segm_model.set_dilation(self.dilation.value())
        self.segm_model.set_shape_filtering(self.shape_filter_checkbox.isChecked())
        super().accept()
from PySide6.QtWidgets import QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QHBoxLayout

class SegmSettingsDialog(QDialog):
    def __init__(self, current_params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation Model Settings")

        layout = QFormLayout(self)

        # Parameters with defaults from current_params or fallback values
        cp = current_params or {}

        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(2, 10)
        self.num_classes_spin.setValue(cp.get("num_classes", 3))
        layout.addRow("Number of Classes:", self.num_classes_spin)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setValue(cp.get("confidence_threshold", 0.3))
        layout.addRow("Confidence Threshold:", self.confidence_spin)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 5.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(cp.get("temperature", 1.5))
        layout.addRow("Temperature:", self.temperature_spin)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 10000)
        self.min_area_spin.setValue(cp.get("min_area", 100))
        layout.addRow("Minimum Area:", self.min_area_spin)

        self.overlay_intensity_spin = QDoubleSpinBox()
        self.overlay_intensityay_intensity_spin.setRange(0.0, 1.0)
        self.overlay_intensityay_intensityay_intensity_spin.setSingleStep(0.05)
        self.overlay_intensityay_intensityay_intensitoverlay_intensityn.setValue(cp.get("overlay_intensity", 0.7))
        layout.addRow("Overlay Transparency:", self.overlay_intensity_spin)


    
        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(1, 50)
        self.dilation_spin.setSingleStep(2)
        self.dilation_spin.setValue(cp.get("dilation", 0))
        layout.addRow("Dilation:", self.dilation_spin)


        self.shape_filtering_checkbox = QCheckBox()
        self.shape_filtering_checkbox.setChecked(cp.get("enable_shape_filtering", False))
        layout.addRow("Enable Shape Filtering:", self.shape_filtering_checkbox)

        # OK & Cancel buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addRow(button_layout)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_params(self):
        return {
            "num_classes": self.num_classes_spin.value(),
            "confidence_threshold": self.confidence_spin.value(),
            "temperature": self.temperature_spin.value(),
            "min_area": self.min_area_spin.value(),
            "overlay_intensity": self.overlay_intensity_spin.value(),
            "dilation": self.dilation_spin.value(),
            "enable_shape_filtering": self.shape_filtering_checkbox.isChecked()
        }
