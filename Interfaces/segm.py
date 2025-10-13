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
            pred = pred.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode='nearest')
            pred = pred.squeeze(0).squeeze(0).to(torch.int64)

        return pred  # [H, W] on self.device

    @torch.inference_mode()
    def create_segmentation_overlay(self, frame_bgr, mask: torch.Tensor, alpha: float = 0.3):
        """
        GPU-friendly overlay of class mask on top of frame.
        - frame_bgr: np.ndarray HxWx3 (BGR) or torch Tensor on any device (H,W,3 or 3,H,W)
        - mask: torch.Tensor [H, W] (int64) on same device as model (preferably CUDA)
        Returns: np.ndarray HxWx3 (BGR) on CPU for display.
        """
        # Normalize frame to a Tensor on model device
        if not isinstance(frame_bgr, torch.Tensor):
            frame = torch.from_numpy(frame_bgr)
        else:
            frame = frame_bgr

        if frame.ndim != 3:
            raise ValueError("Expected frame with 3 dims (HWC or CHW)")

        # Convert to HWC float on device
        if frame.shape[0] in (3, 4):  # CHW
            frame = frame.permute(1, 2, 0)
        frame = frame.to(self.device, dtype=torch.float32, non_blocking=True)

        # If frame channels are RGBA drop A; ensure 3 channels
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Ensure mask on same device and shape matches H,W
        m = mask.to(self.device, dtype=torch.long, non_blocking=True)
        if m.shape[0] != frame.shape[0] or m.shape[1] != frame.shape[1]:
            # Resize nearest on GPU
            m = m.unsqueeze(0).unsqueeze(0).float()
            m = F.interpolate(m, size=(frame.shape[0], frame.shape[1]), mode='nearest')
            m = m.squeeze(0).squeeze(0).to(torch.long)

        # Build a simple class->color LUT on device (BGR)
        # Class 0: transparent (0,0,0); classes 1..num_classes-1 random-ish distinct colors
        num_colors = max(int(self.num_classes), int(m.max().item()) + 1)
        # Fixed palette for determinism
        base_colors = torch.tensor(
            [
                [0, 0, 0],       # background
                [0, 255, 0],     # class 1
                [0, 0, 255],     # class 2
                [255, 0, 0],     # class 3
                [0, 255, 255],   # class 4
                [255, 0, 255],   # class 5
                [255, 255, 0],   # class 6
            ],
            device=self.device,
            dtype=torch.float32,
        )
        if num_colors > base_colors.shape[0]:
            # tile/repeat to cover more classes
            reps = (num_colors + base_colors.shape[0] - 1) // base_colors.shape[0]
            lut = base_colors.repeat((reps, 1))[:num_colors]
        else:
            lut = base_colors[:num_colors]

        # Map mask -> colors on device (H,W,3)
        colored = lut[m.clamp(min=0, max=num_colors - 1)]

        # Alpha blend on device: out = frame*(1-alpha) + colored*alpha, only where mask>0
        alpha_f = float(alpha)
        one_minus = 1.0 - alpha_f
        # Broadcast mask > 0 to 3 channels
        mask3 = (m > 0).unsqueeze(-1)
        out = frame
        out = torch.where(
            mask3,
            (frame * one_minus + colored * alpha_f),
            frame,
        )

        # Clip and convert to uint8 CPU numpy for display
        out = out.clamp(0, 255).to(torch.uint8).cpu().numpy()
        return out