# camera_worker_webcam.py
import time
from itertools import count
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

import cv2

from types_shared import FramePacket


class WebcamWorker(QObject):
    ready = Signal()
    error = Signal(str)
    frame_captured = Signal(object)  # FramePacket

    def __init__(self, width=640, height=640, exposure_us=29000, timeout_ms=1000, device_index=0):
        super().__init__()
        self.width = width
        self.height = height
        self.exposure_us = exposure_us

        self._cap = None  # not persisted; we open-on-demand per snap
        self._live = False
        self._id_counter = count(1)
        self.timeout = timeout_ms
        self.device_index = device_index

    @Slot()
    def initialize(self):
        try:
            # Probe device once to validate availability, but do not keep it open
            cap = cv2.VideoCapture(self.device_index)
            ok = cap.isOpened()
            if ok:
                cap.release()
            if not ok:
                self.error.emit("No webcam device found.")
                return
            self.ready.emit()
        except Exception as e:
            self.error.emit(f"Webcam init failed: {e}")

    @Slot()
    def capture_frame(self):
        """Blocking capture; emits FramePacket."""
        # Open on demand -> flush -> single retrieve -> close (prevents stale frames)
        cap = None
        try:
            cap = cv2.VideoCapture(self.device_index)
            if not cap.isOpened():
                self.error.emit("No webcam device found.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Keep only one buffer if backend supports it
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # Best-effort manual exposure
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual (V4L2)
                cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure_us) / 10000.0)
            except Exception:
                pass

            # Flush stale frames for ~150 ms using grab() (no decode)
            flush_ms = 150
            end_t = time.perf_counter() + (flush_ms / 1000.0)
            while time.perf_counter() < end_t:
                cap.grab()

            t0 = time.perf_counter()
            # Retrieve the freshest frame
            ok, img = cap.retrieve()
            t1 = time.perf_counter()
            if ok and img is not None:
                acq_ms = int((t1 - t0) * 1000)
                packet = FramePacket(image=img, acq_ms=acq_ms, frame_id=next(self._id_counter))
                self.frame_captured.emit(packet)
            else:
                self.error.emit("Webcam read failed.")
        except Exception as e:
            self.error.emit(f"Webcam capture error: {e}")
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    @Slot(int)
    def set_exposure(self, exposure_us: int):
        """Best-effort exposure control for UVC; may be ignored by device."""
        self.exposure_us = int(exposure_us)
        # No persistent cap; exposure will be applied on next capture
        return

    @Slot()
    def stop_live(self):
        self._live = False

    @Slot()
    def shutdown(self):
        try:
            self.stop_live()
            # no persistent capture handle to release
        except Exception:
            pass


