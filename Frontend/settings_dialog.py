from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QWidget
from PySide6.QtCore import Qt

class SettingsDialog(QDialog):
    def __init__(self, segm_params=None, yolo_params=None, cnn_params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.segm_params = segm_params or {}
        self.yolo_params = yolo_params or {}
        self.cnn_params = cnn_params or {}

        self.layout = QVBoxLayout(self)

        # Create Segmentation sliders
        self.segmentation_sliders = {}
        # Store per-parameter conversion so get_params can map back to correct types/ranges
        self._segm_slider_kinds = {}
        self.layout.addWidget(QLabel("Segmentation Settings"))
        for param_name, value in self.segm_params.items():
            slider_widget = self.create_slider(param_name, value)
            self.layout.addWidget(slider_widget)

        # Create YOLO sliders
        self.yolo_sliders = {}
        self._yolo_slider_kinds = {}
        self.layout.addWidget(QLabel("YOLO Confidence Thresholds"))
        for param_name, value in self.yolo_params.items():
            slider_widget = self.create_slider(param_name, value)
            self.layout.addWidget(slider_widget)

        # Create CNN sliders (e.g., classification threshold)
        self.cnn_sliders = {}
        self._cnn_slider_kinds = {}
        if self.cnn_params:
            self.layout.addWidget(QLabel("CNN Classification Settings"))
            for param_name, value in self.cnn_params.items():
                slider_widget = self.create_slider(param_name, value)
                self.layout.addWidget(slider_widget)

        # Buttons
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        self.layout.addLayout(buttons_layout)


    def create_slider(self, name, value):
        container = QWidget()
        layout = QHBoxLayout(container)
        label = QLabel(name)

        slider = QSlider(Qt.Horizontal)
        value_label = QLabel("")

        # Configure different ranges/formatting per-known parameter
        if name == "confidence_threshold":
            # 0.00 .. 1.00 (percentage based)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(int(float(value) * 100))
            value_label.setText(f"{float(value):.2f}")
            slider.valueChanged.connect(lambda val: value_label.setText(f"{val / 100:.2f}"))
            kind = "float01"
        elif name == "temperature":
            # 0.10 .. 5.00 mapped to 10..500
            slider.setMinimum(10)
            slider.setMaximum(500)
            slider.setValue(int(round(float(value) * 100)))
            # Display as float with 2 decimals
            value_label.setText(f"{float(value):.2f}")
            slider.valueChanged.connect(lambda val: value_label.setText(f"{val / 100:.2f}"))
            kind = "float100"
        elif name == "min_area":
            # Integer pixel area, reasonable range 1..10000
            slider.setMinimum(1)
            slider.setMaximum(10000)
            slider.setValue(int(value))
            value_label.setText(str(int(value)))
            slider.valueChanged.connect(lambda val: value_label.setText(str(val)))
            kind = "int"
        else:
            # Default to 0..1 float mapping for unknown segm params
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(int(float(value) * 100))
            value_label.setText(f"{float(value):.2f}")
            slider.valueChanged.connect(lambda val: value_label.setText(f"{val / 100:.2f}"))
            kind = "float01"

        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)

        # Store slider for later retrieval
        if name in self.segm_params:
            self.segmentation_sliders[name] = slider
            self._segm_slider_kinds[name] = kind
        elif name in self.yolo_params:
            # YOLO thresholds: 0.00..1.00
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(int(float(value) * 100))
            # overwrite value label/connect for yolo to ensure correct display
            value_label.setText(f"{float(value):.2f}")
            # Need to disconnect prior lambda? It's fine to keep both for yolo created here
            slider.valueChanged.connect(lambda val: value_label.setText(f"{val / 100:.2f}"))
            self.yolo_sliders[name] = slider
            self._yolo_slider_kinds[name] = "float01"
        else:
            # CNN params: currently handle 'cnn_threshold' as 0..1
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(int(float(value) * 100))
            value_label.setText(f"{float(value):.2f}")
            slider.valueChanged.connect(lambda val: value_label.setText(f"{val / 100:.2f}"))
            self.cnn_sliders[name] = slider
            self._cnn_slider_kinds[name] = "float01"

        return container


    def get_params(self):
        # Map sliders back to param values using the stored kinds
        segm_updated = {}
        for k, slider in self.segmentation_sliders.items():
            kind = self._segm_slider_kinds.get(k, "float01")
            if kind == "int":
                segm_updated[k] = int(slider.value())
            elif kind == "float100":
                segm_updated[k] = slider.value() / 100.0
            else:  # float01
                segm_updated[k] = slider.value() / 100.0

        yolo_updated = {k: slider.value() / 100.0 for k, slider in self.yolo_sliders.items()}
        cnn_updated = {k: slider.value() / 100.0 for k, slider in self.cnn_sliders.items()}
        return segm_updated, yolo_updated, cnn_updated
