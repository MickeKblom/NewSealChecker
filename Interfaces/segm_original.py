from torchvision import transforms  # (you can remove if unused elsewhere)
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import numpy as np

# Use your function; it must output CHW float on same device as input
from ImageProcessing.image_utils import tensor_preprocess
from ImageProcessing.image_utils import morphological_opening

print("SMP version:", smp.__version__)

class SegmentationModel:
    def __init__(
        self,
        model_path,
        num_classes=2,
        confidence_threshold=0.3,
        temperature=1.5,
        min_area=100,
        src_bgr=True,
        device: str | torch.device | None = None,
        input_size=(640, 640),
        return_to_input_size=True,  # if True, upsample mask back to original HxW
    ):
        print(f"Loading model from: {model_path}")
        self.device = torch.device(device) if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        print(f"Using device: {self.device}")

        # Configurable parameters
        self.confidence_threshold = float(confidence_threshold)
        self.temperature = float(temperature)
        self.num_classes = int(num_classes)
        self.min_area = int(min_area)
        self.src_bgr = bool(src_bgr)
        self.input_size = tuple(input_size)
        self.return_to_input_size = bool(return_to_input_size)

        # Initialize model exactly as trained
        self.segm_model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes,
            activation=None,              # raw logits
            decoder_attention_type='scse' # must match training
        )

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        # In newer torch, load_state_dict returns IncompatibleKeys
        incompatible = self.segm_model.load_state_dict(state_dict, strict=True)
        # These prints are helpful during bring-up; remove if noisy
        try:
            print("Missing keys:", incompatible.missing_keys)
            print("Unexpected keys:", incompatible.unexpected_keys)
        except Exception:
            # older torch allowed tuple-unpack; if you use that, keep your original print
            pass

        self.segm_model = self.segm_model.to(self.device).eval()

    @torch.inference_mode()
    def perform_segmentation(self, frame):
        """
        frame: np.ndarray HWC (BGR if from OpenCV) OR torch.Tensor HWC/CHW
        returns: torch.Tensor [H, W] (labels) on self.device
        """
        # Convert to tensor if numpy
        if not isinstance(frame, torch.Tensor):
            frame = torch.from_numpy(frame)
            print("Converted input from numpy -> torch")

        # Save original size (for optional upsampling back)
        if frame.ndim == 3:
            if frame.shape[-1] in (3, 4):  # HWC
                orig_h, orig_w = frame.shape[0], frame.shape[1]
            else:                           # CHW
                orig_h, orig_w = frame.shape[-2], frame.shape[-1]
        else:
            raise ValueError(f"Expected 3D image tensor/array, got shape {tuple(frame.shape)}")

        # Move to device early; tensor_preprocess should keep device
        frame = frame.to(self.device, non_blocking=True)

        # Preprocess to CHW float normalized on device
        x = tensor_preprocess(frame, size=self.input_size, src_bgr=self.src_bgr)  # [C,H,W]
        x = x.unsqueeze(0).contiguous()  # [1,C,H,W]

        # Use modern autocast API; keep disabled on CPU to avoid overhead
        with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
            logits = self.segm_model(x)  # <--- FIXED: call segm_model, not model
            scaled = logits / self.temperature
            probs = torch.softmax(scaled, dim=1)               # [1, num_classes, H, W]
            confidence, predictions = torch.max(probs, dim=1)  # both [1, H, W]

        # Apply confidence threshold (keep on GPU)
        if self.confidence_threshold > 0:
            predictions = predictions.clone()
            predictions[confidence < self.confidence_threshold] = 0

        pred = predictions.squeeze(0).to(torch.int64)  # [H,W], labels

        # Optional GPU-side small-island removal based on min_area via morphology
        # Approximate area filtering using morphological opening with kernel derived from sqrt(min_area)
        if self.min_area > 0:
            # Derive a kernel size k such that k*k ≈ min_area (clamp to odd >=3)
            k = int(torch.sqrt(torch.tensor(float(self.min_area), device=self.device)).item())
            if k < 3:
                k = 3
            if k % 2 == 0:
                k += 1
            # Process each class > 0
            refined = torch.zeros_like(pred)
            for cls in range(1, self.num_classes):
                cls_mask = (pred == cls).float()
                # morphological_opening expects float mask01 on same device
                opened = morphological_opening(cls_mask, kernel_size=k)
                refined = torch.where(opened > 0.5, torch.tensor(cls, dtype=pred.dtype, device=self.device), refined)
            pred = refined

        # Optionally return to original image size
        if self.return_to_input_size and (pred.shape[-2] != orig_h or pred.shape[-1] != orig_w):
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=(orig_h, orig_w), mode='nearest')
            pred = pred.squeeze(0).squeeze(0).to(torch.int64)

        return pred  # [H, W] on self.device

    @torch.inference_mode()
    def create_segmentation_overlay(self, frame_bgr, mask: torch.Tensor, alpha: float = 0.3):
        from Interfaces.segm import SegmentationModel as _dummy  # for signature parity only
        # Reuse original implementation by importing old module in callers if needed.
        raise NotImplementedError


