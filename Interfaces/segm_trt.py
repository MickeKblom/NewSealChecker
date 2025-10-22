import os
import ctypes
import ctypes.util
import time
from typing import Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch

from ImageProcessing.image_utils import tensor_preprocess


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _build_engine_from_onnx(onnx_path: str, input_shape: Tuple[int, int, int, int], fp16: bool = True) -> trt.ICudaEngine:
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 27)  # 128MB
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        inp = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, min=input_shape, opt=input_shape, max=input_shape)
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Engine build failed")
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("Failed to deserialize engine")
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
        self.cudaStreamDestroy = self.lib.cudaStreamDestroy
        self.cudaStreamDestroy.argtypes = [self.c_void_p]
        self.cudaStreamDestroy.restype = self.c_int
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
            raise RuntimeError("Failed to create TensorRT execution context")

        self.uses_new_api = hasattr(self.engine, 'num_io_tensors')
        if self.uses_new_api:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
            self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        else:
            inputs = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
            outputs = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
            self.input_index, self.output_index = inputs[0], outputs[0]
            self.input_name = self.engine.get_binding_name(self.input_index)
            self.output_name = self.engine.get_binding_name(self.output_index)

        self.input_shape = input_shape  # (1,3,H,W)
        if self.uses_new_api and hasattr(self.context, 'set_input_shape'):
            self.context.set_input_shape(self.input_name, self.input_shape)
            out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        else:
            self.context.set_binding_shape(self.input_index, self.input_shape)
            out_shape = tuple(self.context.get_binding_shape(self.output_index))
        if -1 in out_shape:
            # Assume (1,num_classes,H,W)
            out_shape = (1, 2, self.input_shape[2], self.input_shape[3])
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

        print(f"[INFO] TRT Segm runner ready: input {self.input_shape}, output {self.output_shape}, stream=0x{int(self.stream.value):x}")

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


class SegmentationModelTRT:
    def __init__(
        self,
        onnx_path: str = "Models/Segmentation/best.onnx",
        engine_path: str = "Models/Segmentation/best.trt",
        input_size: Tuple[int, int] = (640, 640),
        num_classes: int = 2,
        confidence_threshold: float = 0.3,
        temperature: float = 1.5,
        src_bgr: bool = True,
        return_to_input_size: bool = False,
        fp16: bool = True,
    ) -> None:
        self.input_size = tuple(input_size)
        self.num_classes = int(num_classes)
        self.confidence_threshold = float(confidence_threshold)
        self.temperature = float(temperature)
        self.src_bgr = bool(src_bgr)
        self.return_to_input_size = bool(return_to_input_size)

        # Build or load engine
        input_shape = (1, 3, self.input_size[0], self.input_size[1])
        if engine_path and os.path.exists(engine_path):
            engine = _load_engine(engine_path)
        else:
            engine = _build_engine_from_onnx(onnx_path, input_shape=input_shape, fp16=fp16)
            if engine_path:
                _save_engine(engine, engine_path)
        self.runner = _TrtRunner(engine, input_shape)

    @torch.inference_mode()
    def perform_segmentation(self, frame) -> torch.Tensor:
        # Convert to tensor if numpy
        if not isinstance(frame, torch.Tensor):
            frame = torch.from_numpy(frame)

        # Keep original size to optionally restore
        if frame.ndim == 3:
            if frame.shape[-1] in (3, 4):  # HWC
                orig_h, orig_w = frame.shape[0], frame.shape[1]
            else:  # CHW
                orig_h, orig_w = frame.shape[-2], frame.shape[-1]
        else:
            raise ValueError(f"Expected 3D image tensor/array, got shape {tuple(frame.shape)}")

        # Preprocess (uses same logic as torch model): CHW float normalized
        x_chw = tensor_preprocess(frame, size=self.input_size, src_bgr=self.src_bgr)  # [C,H,W] torch
        x_np = x_chw.unsqueeze(0).contiguous().to(torch.float32).cpu().numpy()  # [1,3,H,W]

        t0 = time.perf_counter()
        y = self.runner.infer(x_np)  # [1,num_classes,H,W] logits float32
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Softmax over classes
        logits = y[0]
        scaled = logits / max(self.temperature, 1e-6)
        # softmax along axis=0 (class axis)
        e = np.exp(scaled - np.max(scaled, axis=0, keepdims=True))
        probs = e / np.sum(e, axis=0, keepdims=True)
        confidence = probs.max(axis=0)       # [H,W]
        predictions = probs.argmax(axis=0)   # [H,W]

        # Confidence threshold -> set to background (0)
        if self.confidence_threshold > 0:
            predictions = predictions.copy()
            predictions[confidence < self.confidence_threshold] = 0

        # Optionally resize back to original size
        pred = predictions
        if self.return_to_input_size and (pred.shape[0] != orig_h or pred.shape[1] != orig_w):
            pred = cv2.resize(pred.astype(np.int32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        print(f"[INFO] TRT Segm inference: {dt_ms:.3f} ms")
        return torch.from_numpy(pred.astype(np.int64))

    @torch.inference_mode()
    def create_segmentation_overlay(self, frame_bgr, mask: torch.Tensor, alpha: float = 0.3):
        """Overlay mask on frame (CPU implementation, API-compatible with original).
        - frame_bgr: np.ndarray HxWx3 (BGR) or torch Tensor (HWC/CHW)
        - mask: torch.Tensor [H,W] (int64)
        Returns: np.ndarray HxWx3 (BGR) uint8
        """
        # Normalize frame to torch HWC float32 on CPU
        if not isinstance(frame_bgr, torch.Tensor):
            frame = torch.from_numpy(frame_bgr)
        else:
            frame = frame_bgr

        if frame.ndim != 3:
            raise ValueError("Expected frame with 3 dims (HWC or CHW)")

        # To HWC
        if frame.shape[0] in (3, 4):
            frame = frame.permute(1, 2, 0)
        frame = frame.to(dtype=torch.float32, device='cpu', non_blocking=False)

        # Ensure 3 channels
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Ensure mask CPU long and resized if needed
        m = mask.to(dtype=torch.long, device='cpu', non_blocking=False)
        if m.shape[0] != frame.shape[0] or m.shape[1] != frame.shape[1]:
            m = torch.from_numpy(cv2.resize(m.cpu().numpy().astype(np.int32), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)).to(torch.long)

        # Simple LUT (BGR)
        num_colors = max(2, int(m.max().item()) + 1)
        base_colors = torch.tensor(
            [
                [0, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [0, 255, 255],
                [255, 0, 255],
                [255, 255, 0],
            ],
            dtype=torch.float32,
        )
        if num_colors > base_colors.shape[0]:
            reps = (num_colors + base_colors.shape[0] - 1) // base_colors.shape[0]
            lut = base_colors.repeat((reps, 1))[:num_colors]
        else:
            lut = base_colors[:num_colors]

        colored = lut[m.clamp(min=0, max=num_colors - 1)]  # H,W,3

        # Alpha blend where mask>0
        alpha_f = float(alpha)
        out = frame
        mask3 = (m > 0).unsqueeze(-1)
        out = torch.where(
            mask3,
            frame * (1.0 - alpha_f) + colored * alpha_f,
            frame,
        )

        out = out.clamp(0, 255).to(torch.uint8).cpu().numpy()
        return out


