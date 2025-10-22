## TensorRT Inference on Jetson (Orin Nano, JetPack 6.x)

This project runs the OD classifier with NVIDIA TensorRT directly from Python on GPU. No PyCUDA, no ONNX Runtime. We use the TensorRT 10.x Python API together with CUDA Runtime (cudart) via `ctypes` for device memory and streams.

### What we implemented

- Minimal TensorRT-only wrapper in `Interfaces/ODPredict.py` that:
  - Builds a TensorRT engine from ONNX or loads a cached engine (`Models/CNN/best.trt`).
  - Fixes the optimization profile to `1x4x224x512` (RGB+mask) matching the pipeline.
  - Allocates GPU input/output buffers with `cudaMalloc` and a non-default `cudaStream` using `ctypes` bindings to `libcudart.so`.
  - Uses TensorRT 10.x API: `set_input_shape`, `set_tensor_address`, `execute_async_v3(stream)`.
  - Preprocesses input like the original model: RGB+mask concat, normalization, and resize to `224x512`.
  - Logs execution time for each inference.

### Why `ctypes` + cudart (not PyCUDA or ONNX Runtime)

- PyCUDA is difficult to install/maintain on JetPack 6.x and required build toolchains (`nvcc`, headers). We avoid it entirely.
- ONNX Runtime TensorRT EP wheels are not readily available for JetPack 6.x on bare-metal; they are mostly supported in NVIDIA containers or require building ORT from source.
- `ctypes` to `libcudart.so` is the simplest reliable path: the CUDA Runtime library ships with JetPack; no extra Python packages.

### Data flow (CPU ↔ GPU)

1. The pipeline provides `image_t` (HWC/CHW) and `mask_t` tensors on CPU.
2. `ODClassifierWrapper._prepare_input(...)` converts to CHW, normalizes, resizes to `224x512`, and returns a CPU `numpy.float32` array with shape `(1,4,224,512)`.
3. `_TrtRunner.infer(x)`
   - Copies host input to device with `cudaMemcpyHostToDevice`.
   - Sets device pointers into the TensorRT execution context via `set_tensor_address`.
   - Calls `execute_async_v3` with a non-default CUDA stream.
   - Copies device output back to host with `cudaMemcpyDeviceToHost`.
   - Synchronizes the stream.

Note: Images start on CPU and are explicitly copied to GPU before inference; outputs are copied back to CPU for downstream logic.

### Segmentation (TRT) specifics

- Wrapper: `Interfaces/segm_trt.py` with `SegmentationModelTRT`.
- Engine IO: `(1,3,640,640)` → `(1,num_classes,640,640)` logits.
- Preprocess: same as Torch model via `tensor_preprocess` (BGR→RGB if configured, normalize, resize to 640x640).
- Postprocess: in Python (softmax, confidence threshold, argmax). Default keeps output at 640x640. Set `return_to_input_size=True` to resize back to source HxW.
- Performance: ~60–120 ms/inference on Orin Nano (FP16, 640x640), after warmup.

### Files

- `Interfaces/ODPredict.py`: TensorRT wrapper and preprocessing.
- `Models/CNN/best.onnx`: ONNX model (input: `1x4x224x512`).
- `Models/CNN/best.trt`: Cached TensorRT engine (created automatically if missing).

### Requirements

- Jetson Orin Nano with JetPack 6.x (CUDA 12.x, TensorRT 10.x present system-wide)
- Python 3.10 (system, not a 3.11 venv; TRT Python bindings match 3.10)
- Python packages from `requirements.txt` (no `tensorrt` pip; it comes with JetPack)

### Setup steps

1. Ensure the ONNX model exists:
   - Place at `Models/CNN/best.onnx` (8–10 MB typical).
2. First run will build the engine (can take minutes on Jetson):
   - The engine is saved to `Models/CNN/best.trt` and loads instantly next time.
3. Run the app with Python 3.10:
   ```bash
   cd /home/ai/Documents/Code/NewSealChecker
   /usr/bin/python3.10 main.py
   ```

### Verifying GPU TensorRT path

- You should see logs like:
  - `TensorRT runner ready: input (1, 4, 224, 512), output (1, 2), stream=0x...`
  - `TensorRT inference: 12.345 ms (logit=..., prob=...)`
  These appear only when GPU buffers and the CUDA stream are used with TensorRT.

### Troubleshooting

- Engine build slow or OOM:
  - We cap workspace to 128 MB. You can increase (e.g., 256–512 MB) if builds fail due to tactic limits.
  - Close memory-heavy apps (IDE/browser) during first build.
- `Cuda Runtime (illegal memory access)`:
  - Ensure we do not pass CPU pointers to TensorRT. This implementation always uses device pointers set via `set_tensor_address`.
- Python version mismatch for TensorRT:
  - Use system Python 3.10 on JetPack 6.x. Avoid 3.11 venvs (TRT wheels are not provided for 3.11).
- ONNX shapes mismatch:
  - We fix the optimization profile at `1x4x224x512` and resize in preprocessing to match.

### Performance notes

- FP16 is enabled when supported; expect faster inference and lower memory use.
- Execution times are printed for each inference (ms). After warmup, it stabilizes.

### Design choices summary

- **TensorRT 10.x Python API**: stable and matches JetPack 6.x.
- **ctypes + cudart**: zero extra dependencies; simple, explicit GPU memory and stream control.
- **Fixed profile**: simplifies context setup and prevents dynamic shape overhead on Jetson.


