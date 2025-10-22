import os
import ctypes
import ctypes.util
import time
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _build_engine_from_onnx(onnx_path: str, input_shape: Tuple[int, int, int, int], fp16: bool = True) -> trt.ICudaEngine:
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    print(f"[INFO] Building YOLO TensorRT engine from {onnx_path} (shape={input_shape}, fp16={fp16}) ...")
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB workspace for YOLO
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        # Keep timing cache disabled to conserve memory
        try:
            config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        except Exception:
            pass

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse YOLO ONNX")

        inp = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, min=input_shape, opt=input_shape, max=input_shape)
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("YOLO engine build failed")
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("Failed to deserialize YOLO engine")
        print("[INFO] YOLO TensorRT engine built successfully")
        return engine


def _load_engine(engine_path: str) -> trt.ICudaEngine:
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load engine: {engine_path}")
    return engine


def _save_engine(engine: trt.ICudaEngine, engine_path: str) -> None:
    try:
        serialized = engine.serialize()
        with open(engine_path, 'wb') as f:
            f.write(serialized)
    except Exception:
        pass


class _CudaRt:
    def __init__(self) -> None:
        libname = ctypes.util.find_library("cudart") or "libcudart.so"
        self.lib = ctypes.CDLL(libname)
        self.c_void_p = ctypes.c_void_p
        self.c_size_t = ctypes.c_size_t
        self.c_int = ctypes.c_int
        self.cudaMalloc = self.lib.cudaMalloc
        self.cudaMalloc.argtypes = [ctypes.POINTER(self.c_void_p), self.c_size_t]
        self.cudaMalloc.restype = self.c_int
        self.cudaFree = self.lib.cudaFree
        self.cudaFree.argtypes = [self.c_void_p]
        self.cudaFree.restype = self.c_int
        self.cudaMemcpy = self.lib.cudaMemcpy
        self.cudaMemcpy.argtypes = [self.c_void_p, self.c_void_p, self.c_size_t, self.c_int]
        self.cudaMemcpy.restype = self.c_int
        self.cudaStreamCreate = self.lib.cudaStreamCreate
        self.cudaStreamCreate.argtypes = [ctypes.POINTER(self.c_void_p)]
        self.cudaStreamCreate.restype = self.c_int
        self.cudaStreamSynchronize = self.lib.cudaStreamSynchronize
        self.cudaStreamSynchronize.argtypes = [self.c_void_p]
        self.cudaStreamSynchronize.restype = self.c_int
        self.cudaMemcpyHostToDevice = 1
        self.cudaMemcpyDeviceToHost = 2


class _TrtRunner:
    def __init__(self, engine: trt.ICudaEngine, input_shape: Tuple[int, int, int, int]):
        self.engine = engine
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create YOLO TRT context")

        self.uses_new_api = hasattr(self.engine, 'num_io_tensors')
        if self.uses_new_api:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
            # Assume single output tensor (export detect layer to output). Adjust if needed.
            self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        else:
            inputs = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
            outputs = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
            self.input_index, self.output_index = inputs[0], outputs[0]
            self.input_name = self.engine.get_binding_name(self.input_index)
            self.output_name = self.engine.get_binding_name(self.output_index)

        self.input_shape = input_shape  # (1,3,640,640)
        if self.uses_new_api and hasattr(self.context, 'set_input_shape'):
            self.context.set_input_shape(self.input_name, self.input_shape)
            out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        else:
            self.context.set_binding_shape(self.input_index, self.input_shape)
            out_shape = tuple(self.context.get_binding_shape(self.output_index))
        # Typical YOLO export gives (1, N, 6) -> [x,y,w,h,conf,cls] per row
        if -1 in out_shape:
            out_shape = (1, 8400, 6)  # fallback; adjust if your model differs
        self.output_shape = out_shape

        self.rt = _CudaRt()
        self.d_input = self.rt.c_void_p()
        self.d_output = self.rt.c_void_p()
        self.input_bytes = int(np.prod(self.input_shape)) * 4
        self.output_bytes = int(np.prod(self.output_shape)) * 4
        r1 = self.rt.cudaMalloc(ctypes.byref(self.d_input), self.input_bytes)
        r2 = self.rt.cudaMalloc(ctypes.byref(self.d_output), self.output_bytes)
        if r1 != 0 or r2 != 0:
            raise RuntimeError(f"cudaMalloc failed: {r1},{r2}")
        self.stream = self.rt.c_void_p()
        r3 = self.rt.cudaStreamCreate(ctypes.byref(self.stream))
        if r3 != 0:
            raise RuntimeError(f"cudaStreamCreate failed: {r3}")

        print(f"[INFO] TRT YOLO runner ready: input {self.input_shape}, output {self.output_shape}, stream=0x{int(self.stream.value):x}")

    def infer(self, x_host: np.ndarray) -> np.ndarray:
        if x_host.dtype != np.float32:
            x_host = x_host.astype(np.float32, copy=False)
        if tuple(x_host.shape) != self.input_shape:
            raise ValueError(f"Expected input {self.input_shape}, got {x_host.shape}")

        y_host = np.empty(self.output_shape, dtype=np.float32)
        rc = self.rt.cudaMemcpy(self.d_input, ctypes.c_void_p(x_host.ctypes.data), self.input_bytes, self.rt.cudaMemcpyHostToDevice)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy HtoD failed: {rc}")
        if self.uses_new_api and hasattr(self.context, 'set_tensor_address'):
            self.context.set_tensor_address(self.input_name, int(self.d_input.value))
            self.context.set_tensor_address(self.output_name, int(self.d_output.value))
            ok = self.context.execute_async_v3(int(self.stream.value))
        else:
            bindings = [0] * self.engine.num_bindings
            bindings[self.input_index] = int(self.d_input.value)
            bindings[self.output_index] = int(self.d_output.value)
            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=int(self.stream.value))
        if not ok:
            raise RuntimeError("TensorRT inference failed")
        rc = self.rt.cudaMemcpy(ctypes.c_void_p(y_host.ctypes.data), self.d_output, self.output_bytes, self.rt.cudaMemcpyDeviceToHost)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy DtoH failed: {rc}")
        self.rt.cudaStreamSynchronize(self.stream)
        return y_host


class YOLOModelTRT:
    def __init__(self, weights_path: str = "Models/YOLO/best.onnx", imgsz: int = 640, fp16: bool = True, classes_path: str = "Models/YOLO/classes.txt") -> None:
        """Drop-in for original YOLOModel.
        - weights_path: can be .onnx or .pt. If .pt, expect a sibling best.onnx in Models/YOLO/.
        """
        self.imgsz = int(imgsz)
        input_shape = (1, 3, self.imgsz, self.imgsz)

        # Resolve ONNX/engine paths from provided weights_path
        onnx_path = weights_path
        if onnx_path.endswith('.pt'):
            # Map .pt to our ONNX default
            onnx_path = os.path.join(os.path.dirname(weights_path) or 'Models/YOLO', 'best.onnx')
        if not onnx_path.endswith('.onnx'):
            # Fallback to default ONNX path
            onnx_path = 'Models/YOLO/best.onnx'
        engine_path = os.path.splitext(onnx_path)[0] + '.trt'

        if os.path.exists(engine_path):
            print(f"[INFO] Loading YOLO TensorRT engine from {engine_path}")
            engine = _load_engine(engine_path)
        else:
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"YOLO ONNX not found: {onnx_path}. Export ONNX first.")
            t0 = time.perf_counter()
            engine = _build_engine_from_onnx(onnx_path, input_shape=input_shape, fp16=fp16)
            t_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[INFO] YOLO engine build time: {t_ms:.0f} ms")
            print(f"[INFO] Saving YOLO engine to {engine_path}")
            _save_engine(engine, engine_path)
        self.runner = _TrtRunner(engine, input_shape)
        # Load class names if available
        self.class_names = {}
        try:
            if classes_path and os.path.exists(classes_path):
                with open(classes_path, 'r', encoding='utf-8') as f:
                    names = [ln.strip() for ln in f.readlines() if ln.strip()]
                    self.class_names = {i: n for i, n in enumerate(names)}
        except Exception:
            self.class_names = {}

    def perform_detection(self, frame, confidence_level: float = 0.25, classes_to_detect=None, iou_threshold: float = 0.45, max_det: int = 300) -> Dict[str, Any]:
        # Convert to numpy HWC
        if isinstance(frame, np.ndarray):
            img = frame
        else:
            import torch
            if isinstance(frame, torch.Tensor):
                if frame.ndim == 3 and frame.shape[0] in (3, 4):
                    img = frame[:3, ...].permute(1, 2, 0).contiguous().to('cpu').numpy()
                elif frame.ndim == 3 and frame.shape[-1] in (3, 4):
                    img = frame[..., :3].to('cpu').numpy()
                else:
                    raise ValueError("Unsupported tensor shape for YOLO input")
            else:
                raise ValueError("Unsupported input type for YOLO input")

        # Preprocess: letterbox to imgsz, BGR->RGB, normalize [0,1], CHW
        img0 = img
        h0, w0 = img0.shape[:2]
        r = min(self.imgsz / h0, self.imgsz / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = self.imgsz - new_unpad[0], self.imgsz - new_unpad[1]
        dw //= 2; dh //= 2
        img_resized = cv2.resize(img0, new_unpad, interpolation=cv2.INTER_LINEAR)
        img_padded = cv2.copyMakeBorder(img_resized, dh, self.imgsz - new_unpad[1] - dh, dw, self.imgsz - new_unpad[0] - dw, cv2.BORDER_CONSTANT, value=(114,114,114))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(img_rgb, (0,1,2)).transpose(2,0,1)  # HWC->CHW
        x = chw[np.newaxis, ...].astype(np.float32)

        t0 = time.perf_counter()
        y = self.runner.infer(x)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        det = y[0]
        # Normalize to (num, C) layout
        det = det.squeeze(0) if det.ndim == 3 else det
        if det.ndim == 2 and det.shape[0] < det.shape[1]:
            det = det.transpose(1, 0)

        # Parse boxes/scores/classes
        boxes_xywh = det[:, :4].astype(np.float32)
        C = det.shape[1]
        if C >= 6:
            obj = det[:, 4].astype(np.float32)
            cls_scores = det[:, 5:].astype(np.float32) if C > 6 else None
            # Heuristic: apply sigmoid if values outside [0,1]
            if obj.max() > 1.0 or obj.min() < 0.0:
                obj = 1.0 / (1.0 + np.exp(-obj))
            if cls_scores is not None:
                if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
                    cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))
                cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
                cls_conf = np.max(cls_scores, axis=1)
                conf = (obj * cls_conf).astype(np.float32)
                cls = cls_ids
            else:
                conf = obj
                cls = np.zeros_like(obj, dtype=np.int32)
            keep = conf >= float(confidence_level)
            # Fallback if zero: lower threshold and/or use objectness only
            if not np.any(keep):
                keep = obj >= max(0.05, float(confidence_level) * 0.5)
                conf = obj
        else:
            # Unexpected format
            print(f"[WARN] YOLO TRT unexpected output channels: {C}")
            return {'boxes': np.zeros((0,4), np.float32), 'scores': np.zeros((0,), np.float32), 'labels': np.zeros((0,), np.int32), 'class_names': {}}

        boxes_xywh = boxes_xywh[keep]
        conf = conf[keep]
        cls = cls[keep]

        # Early exit if nothing passed the threshold(s)
        if boxes_xywh.size == 0:
            print(f"[INFO] TRT YOLO inference: {dt_ms:.3f} ms, 0 detections (all filtered)")
            return {
                'boxes': np.zeros((0, 4), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int32),
                'class_names': {},
            }

        # If boxes look normalized, scale to imgsz
        if boxes_xywh.max() <= 1.5:
            boxes_xywh *= float(self.imgsz)

        # Convert xywh to xyxy in padded image space
        x_c, y_c, w, h = boxes_xywh[:,0], boxes_xywh[:,1], boxes_xywh[:,2], boxes_xywh[:,3]
        x1 = x_c - w/2
        y1 = y_c - h/2
        x2 = x_c + w/2
        y2 = y_c + h/2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Map back to original image coords (undo letterbox)
        gain = r
        boxes_xyxy[:, [0,2]] -= dw
        boxes_xyxy[:, [1,3]] -= dh
        boxes_xyxy /= gain
        boxes_xyxy[:, [0,2]] = boxes_xyxy[:, [0,2]].clip(0, w0 - 1)
        boxes_xyxy[:, [1,3]] = boxes_xyxy[:, [1,3]].clip(0, h0 - 1)

        # Optional class filter
        if classes_to_detect is not None:
            classes_to_detect = set(int(c) for c in classes_to_detect)
            keep_cls = np.array([int(c) in classes_to_detect for c in cls], dtype=bool)
            boxes_xyxy = boxes_xyxy[keep_cls]
            conf = conf[keep_cls]
            cls = cls[keep_cls]

        # NMS
        if boxes_xyxy.shape[0] > 0:
            keep = self._nms(boxes_xyxy, conf, iou_threshold, max_det)
            boxes_xyxy = boxes_xyxy[keep]
            conf = conf[keep]
            cls = cls[keep]

        print(f"[INFO] TRT YOLO inference: {dt_ms:.3f} ms, {boxes_xyxy.shape[0]} detections")
        return {
            'boxes': boxes_xyxy.astype(np.float32),
            'scores': conf.astype(np.float32),
            'labels': cls.astype(np.int32),
            'class_names': self.class_names,
        }

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float, max_det: int) -> np.ndarray:
        if boxes.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1e-3) * (y2 - y1 + 1e-3)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0 and len(keep) < max_det:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-3)
            inds = np.where(iou <= iou_th)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.int64)

    def create_detections_overlay(self, frame, detections: Dict[str, Any]):
        boxes = detections.get('boxes', np.zeros((0, 4), dtype=np.float32))
        scores = detections.get('scores', np.zeros((0,), dtype=np.float32))
        labels = detections.get('labels', np.zeros((0,), dtype=np.int32))
        class_names = detections.get('class_names', {})

        if isinstance(frame, np.ndarray):
            out = frame.copy()
        else:
            out = np.ascontiguousarray(frame)

        COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]

        for (box, score, label) in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(label)
            name = class_names.get(class_id, str(class_id)) if isinstance(class_names, dict) else str(class_id)
            color = COLORS[class_id % len(COLORS)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{name} {score:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out


