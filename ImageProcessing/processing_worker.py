# Acquisition/processing_worker.py
import time
import numpy as np
import torch
from PySide6.QtCore import QObject, Signal, Slot
from ImageProcessing.inference import ImageInferencePipeline
from types_shared import FramePacket, ProcessFlags, ProcessedResult
from device_detect import auto_device



class ProcessingWorker(QObject):
    ready = Signal()
    error = Signal(str)
    processed = Signal(object)  # ProcessedResult

    def __init__(self, segm_model_path, cnn_model_path, yolo_model_path, warmup=True):
        super().__init__()
        self.device_cfg = auto_device()
        self._segm_path = segm_model_path
        self._cnn_path = cnn_model_path
        self._yolo_path = yolo_model_path
        self._warmup = warmup
        self._pipeline = None
        self._live_running = False  # drop-policy flag
        self._segm_params = {
            "confidence_threshold": 0.3,
            "temperature": 1.5,
            "min_area": 100,
        }
        self._cnn_params = {
            "cnn_threshold": None,  # None => use checkpoint default
            "debug_show_inputs": False,
        }

    @Slot()
    def initialize(self):
        try:
            # Initialize models ON THIS THREAD (critical for GPU contexts)
            self._pipeline = ImageInferencePipeline(
                segm_model_path=self._segm_path,
                cnn_model_path=self._cnn_path,
                yolo_model_path=self._yolo_path,
                device_cfg=self.device_cfg
            )
            # Apply initial segmentation params
            try:
                m = self._pipeline.segm_model
                m.confidence_threshold = float(self._segm_params["confidence_threshold"])
                m.temperature = float(self._segm_params["temperature"])
                m.min_area = int(self._segm_params["min_area"])
            except Exception:
                pass

            if self._warmup:
                # Warm-up to load kernels/JIT and reduce first-frame latency
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self._pipeline.process_image(
                    dummy, predict_segm=False, predict_class=False, predict_yolo=False
                )
            self.ready.emit()
        except Exception as e:
            self.error.emit(f"Pipeline init failed: {e}")

    @Slot(object, object)
    def process(self, frame_packet: FramePacket, flags: ProcessFlags):
        if self._pipeline is None:
            self.error.emit("Pipeline not initialized.")
            return

        # Live mode: drop if already computing
        if flags.live and self._live_running:
            return

        self._live_running = True
        t0 = time.perf_counter()
        try:
            results = self._pipeline.process_image(
                frame_packet.image,
                predict_segm=flags.predict_segm,
                predict_class=flags.predict_class,
                predict_yolo=flags.predict_yolo,
                postprocess_enabled=False,
            )

            overlay = frame_packet.image.copy()
            if results.get('segmentation_mask') is not None:
                overlay = self._pipeline.segm_model.create_segmentation_overlay(
                    overlay, results['segmentation_mask']
                )
            if results.get('yolo_detections') is not None:
                overlay = self._pipeline.yolo_model.create_detections_overlay(
                    overlay, results['yolo_detections']
                )

            # If debug enabled and cropped inputs are present, attach previews (downscaled for UI)
            if self._cnn_params.get("debug_show_inputs") and results.get('cnn_input_img') is not None:
                try:
                    # Ensure on CPU uint8 for safe Qt display
                    dbg_img = (results['cnn_input_img'].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                    dbg_mask = (results['cnn_input_mask'] > 0).to(torch.uint8).cpu().numpy()
                    results['cnn_input_img_np'] = dbg_img
                    results['cnn_input_mask_np'] = dbg_mask
                except Exception:
                    pass

            t1 = time.perf_counter()
            proc_ms = int((t1 - t0) * 1000) 

            out = ProcessedResult(
                raw_bgr=frame_packet.image,
                overlay_bgr=overlay,
                results=results,
                acq_ms=frame_packet.acq_ms,
                proc_ms=proc_ms,
                frame_id=frame_packet.frame_id,
            )
            self.processed.emit(out)
        except Exception as e:
            self.error.emit(f"Inference failed: {e}")
        finally:
            self._live_running = False

    @Slot()
    def shutdown(self):
        # If your frameworks require explicit cleanup, do it here.
        pass

    @Slot(dict)
    def update_segm_params(self, params: dict):
        """Update segmentation thresholds and propagate to model on the processing thread."""
        self._segm_params.update(params or {})
        try:
            m = self._pipeline.segm_model
            m.confidence_threshold = float(self._segm_params.get("confidence_threshold", m.confidence_threshold))
            m.temperature = float(self._segm_params.get("temperature", m.temperature))
            m.min_area = int(self._segm_params.get("min_area", m.min_area))
        except Exception:
            pass

    @Slot(dict)
    def update_cnn_params(self, params: dict):
        self._cnn_params.update(params or {})
        # Apply threshold override to classifier if provided
        try:
            thr = self._cnn_params.get("cnn_threshold", None)
            if thr is not None and self._pipeline is not None:
                self._pipeline.classifier.threshold = float(thr)
        except Exception:
            pass