import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


class ODClassificationCNN(nn.Module):
    """MobileNetV2 for OD region classification with 4-channel input (RGB + mask)."""
    def __init__(self, num_classes=2):
        super(ODClassificationCNN, self).__init__()
        # Load pretrained MobileNetV2 weights
        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Modify first conv layer to accept 4 channels instead of 3
        original_conv = self.mobilenet.features[0][0]
        new_conv = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        # Copy pretrained weights for first 3 channels; init 4th channel as mean of existing weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight
            new_conv.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
        self.mobilenet.features[0][0] = new_conv

        # Replace classifier for binary classification
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


class ODClassifierWrapper:
    def __init__(self, model_path, device=None):
        self.device = device if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = ODClassificationCNN(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Normalization adapted for RGB+mask channel
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(-1, 1, 1).to(self.device)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225, 0.25]).view(-1, 1, 1).to(self.device)

    def dilate_mask(self, mask, kernel_size=3, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask, kernel, iterations=iterations)
        return dilated

    def prepare_input(self, image_pil, mask_pil):
        # Convert mask to numpy and dilate it
        mask_np = np.array(mask_pil)
        mask_np = self.dilate_mask(mask_np, kernel_size=3, iterations=1)
        mask_np = (mask_np > 0).astype(np.float32)  # Ensure binary

        # Convert image to tensor (3,H,W)
        image_tensor = transforms.ToTensor()(image_pil)
        # Mask tensor (1,H,W)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        # Concatenate to 4 channels (4,H,W)
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=0)

        # Normalize channels separately
        input_tensor = (input_tensor.to(self.device) - self.normalize_mean) / self.normalize_std

        # Add batch dimension (1,4,H,W)
        return input_tensor.unsqueeze(0)

    def predict(self, image_pil, mask_pil):
        """
        Given a cropped RGB PIL image and corresponding binary mask PIL image,
        returns the predicted class index.
        """
        input_tensor = self.prepare_input(image_pil, mask_pil)
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class


