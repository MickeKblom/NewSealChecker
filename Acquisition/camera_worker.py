# camera_worker.py
import time
from itertools import count
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QTimer, Qt
import imagingcontrol4 as im4

from types_shared import FramePacket

class CameraWorker(QObject):
    ready = Signal()
    error = Signal(str)
    frame_captured = Signal(object)  # FramePacket

    def __init__(self, width=640, height=640, exposure_us=29000, timeout_ms=1000):
        super().__init__()
        self.width = width
        self.height = height
        self.exposure_us = exposure_us

        self._grabber = None
        self._sink = None
        self._live = False
        self._id_counter = count(1)
        self.timeout = timeout_ms

    @Slot()
    def initialize(self):
        try:
            im4.Library.init()
            grabber = im4.Grabber()
            devices = im4.DeviceEnum.devices()
            if not devices:
                raise RuntimeError("No camera devices found.")
            grabber.device_open(devices[0])
            pm = grabber.device_property_map
            pm.set_value(im4.PropId.PIXEL_FORMAT, im4.PixelFormat.BGR8)
            pm.set_value(im4.PropId.WIDTH, self.width)
            pm.set_value(im4.PropId.HEIGHT, self.height)
            pm.set_value(im4.PropId.EXPOSURE_TIME, self.exposure_us)
            sink = im4.SnapSink()
            grabber.stream_setup(sink, setup_option=im4.StreamSetupOption.ACQUISITION_START)

            self._grabber = grabber
            self._sink = sink
            self.ready.emit()
        except Exception as e:
            self.error.emit(f"Camera init failed: {e}")

    @Slot()
    def capture_frame(self):
        """Blocking snap in camera thread; emits FramePacket."""
        if not self._sink:
            self.error.emit("Camera not initialized.")
            return
        t0 = time.perf_counter()
        frame = self._sink.snap_single(self.timeout)
        t1 = time.perf_counter()
        if frame:
            # Detach from SDK’s memory to pass to other threads
            img = frame.numpy_wrap().copy()
            acq_ms = int((t1 - t0) * 1000)
            packet = FramePacket(image=img, acq_ms=acq_ms, frame_id=next(self._id_counter))
            self.frame_captured.emit(packet)
        else:
            self.error.emit("Snap timeout.")

    @Slot(int)
    def set_exposure(self, exposure_us: int):
        """Keep your working exposure approach—just run it here."""
        try:
            if self._grabber is None:
                self.error.emit("Camera not initialized.")
                return
            pm = self._grabber.device_property_map
            exposure_auto_prop = pm['Exposure_Auto']
            exposure_auto_prop.value = False
            exposure_time_prop = pm['ExposureTime']
            exposure_time_prop.value = int(exposure_us)
            self.exposure_us = int(exposure_us)
        except Exception as e:
            self.error.emit(f"Failed to set exposure: {e}")



    @Slot()
    def shutdown(self):
        try:
            self.stop_live()
            if self._grabber:
                try:
                    self._grabber.stream_stop()
                except Exception:
                    pass
                self._grabber.device_close()
            im4.Library.close()
        except Exception:
            pass