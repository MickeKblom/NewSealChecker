from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QWidget
from PySide6.QtCore import Qt

class SettingsDialog(QDialog):
    def __init__(self, segm_params=None, yolo_params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.segm_params = segm_params or {}
        self.yolo_params = yolo_params or {}

        self.layout = QVBoxLayout(self)

        # Create Segmentation sliders
        self.segmentation_sliders = {}
        self.layout.addWidget(QLabel("Segmentation Settings"))
        for param_name, value in self.segm_params.items():
            slider_widget = self.create_slider(param_name, value)
            self.layout.addWidget(slider_widget)

        # Create YOLO sliders
        self.yolo_sliders = {}
        self.layout.addWidget(QLabel("YOLO Confidence Thresholds"))
        for param_name, value in self.yolo_params.items():
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
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(int(value * 100))

        # Label for showing the current value of the slider (converted to float with two decimals)
        value_label = QLabel(f"{value:.2f}")

        # When slider value changes, update the value_label to show current value in float form
        slider.valueChanged.connect(lambda val: value_label.setText(f"{val / 100:.2f}"))

        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_label)

        # Store slider for later retrieval
        if name in self.segm_params:
            self.segmentation_sliders[name] = slider
        else:
            self.yolo_sliders[name] = slider

        return container


    def get_params(self):
        segm_updated = {k: slider.value() / 100 for k, slider in self.segmentation_sliders.items()}
        yolo_updated = {k: slider.value() / 100 for k, slider in self.yolo_sliders.items()}
        return segm_updated, yolo_updated
