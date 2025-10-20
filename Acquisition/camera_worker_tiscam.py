# camera_worker_tiscam.py
import time
import sys
import os
from itertools import count
from typing import Optional
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

# GStreamer / TIS camera stack
_GI_IMPORTED = False
_IMPORT_ERR: Optional[str] = None

def _ensure_gi():
    global _GI_IMPORTED, _IMPORT_ERR
    if _GI_IMPORTED:
        return
    try:
        import gi  # type: ignore
        gi.require_version('Gst', '1.0')
        try:
            gi.require_version('Tcam', '0.1')
        except ValueError:
            try:
                gi.require_version('Tcam', '1.0')
            except ValueError:
                pass
        from gi.repository import Gst, Tcam  # type: ignore
        # Initialize GStreamer once per process (safe to call multiple times)
        Gst.init(None)
        _GI_IMPORTED = True
        return Gst, Tcam
    except Exception as e:
        # Try to add common system dist-packages paths (useful when running inside a venv)
        added = False
        candidate_paths = [
            '/usr/lib/python3/dist-packages',
            f"/usr/lib/python{sys.version_info.major}/dist-packages",
            f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages",
        ]
        for p in candidate_paths:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)
                added = True
        # Also make sure GI can find typelibs and shared libs
        gi_typelib_candidates = [
            '/usr/lib/girepository-1.0',
            '/usr/lib/aarch64-linux-gnu/girepository-1.0',
            '/usr/lib/x86_64-linux-gnu/girepository-1.0',
        ]
        ld_lib_candidates = [
            '/usr/lib',
            '/usr/lib/aarch64-linux-gnu',
            '/usr/lib/x86_64-linux-gnu',
        ]
        gi_path = os.environ.get('GI_TYPELIB_PATH', '')
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        for p in gi_typelib_candidates:
            if os.path.isdir(p) and p not in gi_path:
                gi_path = (gi_path + os.pathsep + p) if gi_path else p
        for p in ld_lib_candidates:
            if os.path.isdir(p) and p not in ld_path:
                ld_path = (ld_path + os.pathsep + p) if ld_path else p
        if gi_path:
            os.environ['GI_TYPELIB_PATH'] = gi_path
        if ld_path:
            os.environ['LD_LIBRARY_PATH'] = ld_path
        # If gi partially imported, clean it before retrying
        if 'gi' in sys.modules:
            try:
                del sys.modules['gi']
            except Exception:
                pass
        if added:
            try:
                import gi  # type: ignore
                gi.require_version('Gst', '1.0')
                try:
                    gi.require_version('Tcam', '0.1')
                except ValueError:
                    try:
                        gi.require_version('Tcam', '1.0')
                    except ValueError:
                        pass
                from gi.repository import Gst, Tcam  # type: ignore
                Gst.init(None)
                _GI_IMPORTED = True
                return Gst, Tcam
            except Exception as e2:
                _IMPORT_ERR = f"PyGObject/GStreamer not available after sys.path fix: {e2}"
                raise
        _IMPORT_ERR = f"PyGObject/GStreamer not available: {e}"
        raise

from types_shared import FramePacket


class CameraWorker(QObject):
    ready = Signal()
    error = Signal(str)
    frame_captured = Signal(object)  # FramePacket

    def __init__(self, width: int = 640, height: int = 640, exposure_us: int = 29000, timeout_ms: int = 1000):
        super().__init__()
        self.width = width
        self.height = height
        self.exposure_us = exposure_us
        self.timeout = timeout_ms

        self._pipeline: Optional[object] = None
        self._source: Optional[object] = None  # tcambin
        self._appsink: Optional[object] = None
        self._id_counter = count(1)

        # Defer GI imports to runtime to avoid import-time crashes
        self._Gst = None
        self._Tcam = None

    def _emit_error(self, message: str) -> None:
        self.error.emit(message)

    def _create_pipeline(self):
        Gst = self._Gst
        pipeline = Gst.Pipeline.new("tiscam-pipeline")

        source = Gst.ElementFactory.make("tcambin", "source")
        if source is None:
            raise RuntimeError("Failed to create tcambin. Ensure tiscamera is installed.")

        # Convert and enforce BGR output for downstream consumers
        convert = Gst.ElementFactory.make("videoconvert", "convert")
        if convert is None:
            raise RuntimeError("Failed to create videoconvert.")

        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        if capsfilter is None:
            raise RuntimeError("Failed to create capsfilter.")
        # Request BGR with specific width/height; framerate left unspecified
        caps_str = f"video/x-raw,format=BGR,width={self.width},height={self.height}"
        caps = Gst.Caps.from_string(caps_str)
        capsfilter.set_property("caps", caps)

        appsink = Gst.ElementFactory.make("appsink", "sink")
        if appsink is None:
            raise RuntimeError("Failed to create appsink.")
        appsink.set_property("emit-signals", False)
        appsink.set_property("sync", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)

        for elem in (source, convert, capsfilter, appsink):
            pipeline.add(elem)
        if not Gst.Element.link_many(source, convert, capsfilter, appsink):
            raise RuntimeError("Failed to link GStreamer elements for tiscamera pipeline.")

        self._source = source
        self._appsink = appsink
        return pipeline

    def _configure_exposure(self, exposure_us: int) -> None:
        if self._source is None:
            raise RuntimeError("Camera not initialized.")

        # tcambin is a GstBin; query child implementing Tcam.PropertyProvider
        provider = None
        Gst = self._Gst
        Tcam = self._Tcam
        if isinstance(self._source, Gst.Bin):
            provider = self._source.get_by_interface(Tcam.PropertyProvider.__gtype__)
        if provider is None:
            # Fallback: try casting source itself (older versions)
            try:
                provider = self._source  # type: ignore[assignment]
            except Exception:
                pass

        if provider is None or not hasattr(provider, 'set_tcam_property'):
            raise RuntimeError("TIS property provider not available. Check tiscamera installation.")

        # Disable auto exposure, then set manual exposure time in microseconds
        try:
            provider.set_tcam_property("Exposure Auto", False)
        except Exception:
            # Some older stacks might use different naming
            try:
                provider.set_tcam_property("Exposure_Auto", False)
            except Exception as e:
                raise RuntimeError(f"Failed to disable auto exposure: {e}")

        # Try common property names for exposure time (µs)
        exposure_set = False
        for name in ("Exposure Time (us)", "ExposureTime", "Exposure Time"):
            try:
                provider.set_tcam_property(name, int(exposure_us))
                exposure_set = True
                break
            except Exception:
                continue
        if not exposure_set:
            raise RuntimeError("Failed to set exposure time; unsupported property name.")

    @Slot()
    def initialize(self):
        try:
            try:
                Gst, Tcam = _ensure_gi()
                self._Gst = Gst
                self._Tcam = Tcam
            except Exception:
                raise RuntimeError(_IMPORT_ERR or "PyGObject not available")

            pipeline = self._create_pipeline()

            # Start streaming, then configure exposure once the device is opened
            ret = pipeline.set_state(self._Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start GStreamer pipeline for tiscamera.")

            # Wait shortly to ensure the source is ready before property access
            # (avoids race on slow devices)
            bus = pipeline.get_bus()
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 1.0:
                msg = bus.timed_pop_filtered(10 * self._Gst.MSECOND, self._Gst.MessageType.STATE_CHANGED)
                if msg is None:
                    continue
                if msg.src == pipeline and msg.type == self._Gst.MessageType.STATE_CHANGED:
                    new_state = msg.parse_state_changed()[1]
                    if new_state == self._Gst.State.PLAYING:
                        break

            self._pipeline = pipeline

            # Configure exposure after pipeline is live
            try:
                self._configure_exposure(self.exposure_us)
            except Exception as e:
                # Not fatal for initialization; report but continue
                self._emit_error(f"Exposure setup warning: {e}")

            self.ready.emit()
        except Exception as e:
            self._emit_error(f"Camera init failed: {e}")

    @Slot()
    def capture_frame(self):
        """Blocking snap in camera thread; emits FramePacket."""
        if self._appsink is None:
            self._emit_error("Camera not initialized.")
            return

        t0 = time.perf_counter()
        # try_pull_sample expects timeout in nanoseconds
        timeout_ns = int(self.timeout * 1_000_000)
        sample = self._appsink.emit("try-pull-sample", timeout_ns)
        t1 = time.perf_counter()

        if sample is None:
            self._emit_error("Snap timeout.")
            return

        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        width = s.get_value("width")
        height = s.get_value("height")
        try:
            success, map_info = buf.map(self._Gst.MapFlags.READ)
            if not success:
                self._emit_error("Failed to map buffer.")
                return
            try:
                # appsink caps enforce BGR, 3 channels
                array = np.frombuffer(map_info.data, dtype=np.uint8)
                img = array.reshape((height, width, 3)).copy()
            finally:
                buf.unmap(map_info)
        except Exception as e:
            self._emit_error(f"Failed to convert frame: {e}")
            return

        acq_ms = int((t1 - t0) * 1000)
        packet = FramePacket(image=img, acq_ms=acq_ms, frame_id=next(self._id_counter))
        self.frame_captured.emit(packet)

    @Slot(int)
    def set_exposure(self, exposure_us: int):
        try:
            self._configure_exposure(int(exposure_us))
            self.exposure_us = int(exposure_us)
        except Exception as e:
            self._emit_error(f"Failed to set exposure: {e}")

    @Slot()
    def shutdown(self):
        try:
            if self._pipeline is not None:
                self._pipeline.set_state(self._Gst.State.NULL)
        except Exception:
            pass


def is_tiscam_available() -> bool:
    """Return True if GI, GStreamer and Tcam are importable and tcambin exists."""
    try:
        Gst, Tcam = _ensure_gi()
    except Exception:
        return False
    try:
        # Sanity check: make sure tcambin element is available
        factory = Gst.ElementFactory.find("tcambin")
        return factory is not None
    except Exception:
        return False


