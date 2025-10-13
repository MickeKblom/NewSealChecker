import imagingcontrol4 as im4
import cv2
from PySide6.QtCore import QObject, Signal
import numpy as np
from PySide6.QtCore import QTimer

class CameraApp(QObject):
    new_frame = Signal(np.ndarray)  # Signal to emit new frames (BGR numpy array)

    def __init__(self):
        super().__init__()
        self.camera, self.sink = self.init_cam()
        self.live_mode = False

    def init_cam(self):
        im4.Library.init()
        grabber = im4.Grabber()
        devices = im4.DeviceEnum.devices()
        if not devices:
            print("No devices found.")
            return None, None
        grabber.device_open(devices[0])
        grabber.device_property_map.set_value(im4.PropId.PIXEL_FORMAT, im4.PixelFormat.BGR8)
        grabber.device_property_map.set_value(im4.PropId.WIDTH, 640)
        grabber.device_property_map.set_value(im4.PropId.HEIGHT, 640)
        grabber.device_property_map.set_value(im4.PropId.EXPOSURE_TIME, 29000)
        sink = im4.SnapSink()
        grabber.stream_setup(sink, setup_option=im4.StreamSetupOption.ACQUISITION_START)
        return grabber, sink

    def capture_single_image(self):
        if not self.sink:
            return None
        frame = self.sink.snap_single(1000)
        if frame:
            return frame.numpy_wrap()
        return None

    def start_live_mode(self):
        self.live_mode = True
        self._update_live()

    def stop_live_mode(self):
        self.live_mode = False

    def _update_live(self):
        if not self.live_mode or not self.sink:
            return
        frame = self.sink.snap_single(1000)
        if frame:
            img = frame.numpy_wrap()
            self.new_frame.emit(img)
        # Schedule next update in 30ms
        
        QTimer.singleShot(30, self._update_live)


    def set_exposure(self, value):
        """ Set exposure time with auto-exposure disabled using correct property. """
        exposure_time = int(value)
        if self.camera:
            try:
                # Disable auto-exposure using the correct property
                exposure_auto_prop = self.camera.device_property_map['Exposure_Auto']
                exposure_auto_prop.value = False  # Use .value assignment
                # Now set the exposure time
                exposure_time_prop = self.camera.device_property_map['ExposureTime']
                exposure_time_prop.value = exposure_time  # Use .value assignment
                print(f"Exposure set to {exposure_time} µs (manual mode, auto-exposure OFF)")
                self.current_exposure = exposure_time
            except Exception as e:
                print(f"Failed to set exposure: {e}")
        else:
            print("Camera not initialized!")
