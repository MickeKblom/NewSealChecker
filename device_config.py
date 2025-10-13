from enum import Enum
from dataclasses import dataclass

class Backend(Enum):
    TORCH = "torch"
    ORT = "onnxruntime"
    TRT = "tensorrt"

@dataclass
class DeviceConfig:
    backend: Backend
    device: str
    fp16: bool = False  # Optional: for half-precision inference