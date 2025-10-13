# types_shared.py
from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, Optional

@dataclass
class FramePacket:
    image: np.ndarray               # BGR uint8
    acq_ms: float
    frame_id: int

@dataclass
class ProcessFlags:
    live: bool                      # helps the processor decide to drop/skip
    predict_segm: bool = False
    predict_class: bool = False
    predict_yolo: bool = False
    postprocess_mask: bool = False

@dataclass
class ProcessedResult:
    raw_bgr: np.ndarray         # original frame
    overlay_bgr: np.ndarray
    results: Dict[str, Any]
    acq_ms: float
    proc_ms: float
    frame_id: int