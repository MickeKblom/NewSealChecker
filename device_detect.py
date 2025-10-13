# device_detect.py
from device_config import DeviceConfig, Backend

def auto_device() -> DeviceConfig:
    try:
        import torch
        if torch.cuda.is_available():
            return DeviceConfig(backend=Backend.TORCH, device='cuda:0', fp16=False)
    except Exception:
        pass

    # Try ONNX Runtime with preferred GPU providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'TensorrtExecutionProvider' in providers:
            return DeviceConfig(backend=Backend.ORT, device='TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in providers:
            return DeviceConfig(backend=Backend.ORT, device='CUDAExecutionProvider')
        # Windows GPU (AMD/Intel/NVIDIA) via DirectML
        if 'DmlExecutionProvider' in providers:
            return DeviceConfig(backend=Backend.ORT, device='DmlExecutionProvider')
        ## Fallback CPU
        #return DeviceConfig(backend=Backend.ORT, device='CPUExecutionProvider')
    except Exception:
        pass

    # Last resort
    return DeviceConfig(backend=Backend.TORCH, device='cpu')