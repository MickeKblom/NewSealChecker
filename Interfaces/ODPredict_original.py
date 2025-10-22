import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights


class ODClassificationCNN(nn.Module):
    """MobileNetV2 backbone adapted for 4-channel input (RGB + mask) with single-logit output."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.mobilenet = models.mobilenet_v2(weights=weights)

        # Replace stem conv: 3 → 4 channels
        original_conv = self.mobilenet.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        with torch.no_grad():
            if weights is not None:
                new_conv.weight[:, :3, :, :] = original_conv.weight
                new_conv.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new_conv.weight, nonlinearity='relu')
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
        self.mobilenet.features[0][0] = new_conv

        # Single-logit head for binary classification
        in_features = self.mobilenet.last_channel
        self.mobilenet.classifier[1] = nn.Linear(in_features, 1)
        nn.init.normal_(self.mobilenet.classifier[1].weight, std=0.01)
        nn.init.zeros_(self.mobilenet.classifier[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mobilenet(x)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Load model and optional operating threshold from a checkpoint.
        Expects keys: 'model_state_dict' and optionally 'thr_at_recall1'.
        Returns: (model, threshold_float)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(ckpt_path, map_location=device)
        model = cls(pretrained=False)
        if dtype is not None:
            model = model.to(dtype)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        thr = float(ckpt.get("thr_at_recall1", 0.5))
        return model, thr


class ODClassifierWrapper:
    """Pipeline-friendly wrapper: loads 4-channel MobileNetV2 single-logit model and predicts from torch tensors.
    Expected inputs from pipeline:
      - image_t: torch.Tensor HxWx3 or 3xHxW, uint8 or float [0..1]
      - mask_t:  torch.Tensor HxW or 1xHxW, integer/uint8/float (nonzero => foreground)
    """
    def __init__(self, checkpoint_path: str, device: torch.device | None = None, onnx_path: str | None = "Models/CNN/best.onnx"):
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model, self.threshold = ODClassificationCNN.from_checkpoint(checkpoint_path, device=self.device)

        # Channel-wise normalization (RGB ImageNet, mask centered at 0.5)
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406, 0.5], device=self.device).view(-1, 1, 1)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225, 0.25], device=self.device).view(-1, 1, 1)

        # Optional ONNX Runtime session (prefer TensorRT EP when available)
        self.ort_session = None
        self.ort_input_name = None
        try:
            import onnxruntime as ort  # type: ignore
            if onnx_path is not None:
                providers = []
                try:
                    # Prefer TensorRT, then CUDA, then CPU
                    providers = [
                        'TensorrtExecutionProvider',
                        'CUDAExecutionProvider',
                        'CPUExecutionProvider',
                    ]
                    available = ort.get_available_providers()
                    providers = [p for p in providers if p in available]
                except Exception:
                    providers = ['CPUExecutionProvider']

                so = ort.SessionOptions()
                self.ort_session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
                self.ort_input_name = self.ort_session.get_inputs()[0].name
        except Exception:
            # ONNX Runtime not available or session init failed; will fall back to Torch
            self.ort_session = None

    def _prepare_input_from_tensors(self, image_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        # Move to device
        img = image_t.to(self.device, non_blocking=True)
        m = mask_t.to(self.device, non_blocking=True)

        # Ensure shapes to CHW
        if img.ndim != 3:
            raise ValueError("image_t must have 3 dims (HWC or CHW)")
        if img.shape[0] in (3, 4):
            img_chw = img
        else:
            # HWC -> CHW
            img_chw = img.permute(2, 0, 1).contiguous()

        # Scale to float [0,1] if integer type
        if not torch.is_floating_point(img_chw):
            img_chw = img_chw.float() / 255.0

        # Mask to 1xHxW float in {0,1}
        if m.ndim == 2:
            mask_1hw = m.unsqueeze(0)
        elif m.ndim == 3 and m.shape[0] == 1:
            mask_1hw = m
        else:
            # If HWC, squeeze channel or convert
            if m.ndim == 3 and m.shape[-1] == 1:
                mask_1hw = m.permute(2, 0, 1)
            else:
                mask_1hw = m.unsqueeze(0)
        mask_1hw = (mask_1hw > 0).to(img_chw.dtype)

        # Concatenate RGB + mask
        if img_chw.shape[0] < 3:
            raise ValueError("image_t must contain 3 color channels")
        x4 = torch.cat([img_chw[:3, ...], mask_1hw], dim=0)

        # Normalize
        x4 = (x4 - self.normalize_mean) / self.normalize_std
        return x4.unsqueeze(0)  # 1,4,H,W

    @torch.inference_mode()
    def predict(self, image_t: torch.Tensor, mask_t: torch.Tensor) -> int:
        # If an ORT session is available, use it for faster inference on Jetson (TensorRT/CUDA/CPU), else Torch
        if self.ort_session is not None and self.ort_input_name is not None:
            x = self._prepare_input_from_tensors(image_t, mask_t)
            # ORT expects CPU numpy float32
            x_np = x.detach().to('cpu', dtype=torch.float32, non_blocking=True).numpy()
            outputs = self.ort_session.run(None, {self.ort_input_name: x_np})
            # Assume single output logits (1,1)
            import numpy as np
            logits = np.array(outputs[0]).reshape(-1)
            prob = 1.0 / (1.0 + np.exp(-logits[0]))
            return int(float(prob) >= float(self.threshold))
        else:
            x = self._prepare_input_from_tensors(image_t, mask_t)
            logits = self.model(x)  # (1,1)
            prob = torch.sigmoid(logits.squeeze())
            return int((prob >= self.threshold).item())


