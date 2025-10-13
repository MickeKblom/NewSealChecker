from PySide6.QtCore import QObject, Signal
import numpy as np
from PySide6.QtCore import QTimer

class CameraApp(QObject):
    new_frame = Signal(np.ndarray)  # Signal to emit new frames (BGR numpy array)

    def __init__(self, camera_worker):
        super().__init__()
        self._worker = camera_worker
        self._worker.frame_captured.connect(self._emit_frame)
        self.live_mode = False

    def capture_single_image(self):
        self._worker.capture_frame()
        # Frame will be emitted asynchronously via frame_captured -> new_frame
        return None

    def start_live_mode(self):
        self.live_mode = True
        self._update_live()

    def stop_live_mode(self):
        self.live_mode = False

    def _update_live(self):
        if not self.live_mode:
            return
        self._worker.capture_frame()
        QTimer.singleShot(30, self._update_live)


    def set_exposure(self, value):
        self._worker.set_exposure(int(value))

    def _emit_frame(self, packet_or_img):
        # Support both raw ndarray and FramePacket(image=...)
        if isinstance(packet_or_img, np.ndarray):
            self.new_frame.emit(packet_or_img)
            return
        try:
            self.new_frame.emit(packet_or_img.image)
        except Exception:
            pass
