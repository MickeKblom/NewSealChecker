from torchvision import transforms
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter
from skimage.measure import label, regionprops
import segmentation_models_pytorch as smp
import torch
from PIL import Image
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.measure import find_contours
import matplotlib.pyplot as plt
print("SMP version:", smp.__version__)
class SegmentationModel:
    def __init__(self, model_path, num_classes=2, confidence_threshold=0.3, temperature=1.5, 
                 min_area=100, morph_kernel_size=3, overlay_intensity=0.7,
                 enable_shape_filtering=False, min_aspect_ratio=0.5, max_aspect_ratio=10.0,
                 min_curvature=0.0, max_curvature=5.0, enable_crf=True, enable_ellipse_fitting=False,
                 enable_banana_processing=False, dilation=0):
        print(f"Loading model from: {model_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Configurable parameters
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.num_classes = num_classes
        self.min_area = min_area
        self.morph_kernel_size = morph_kernel_size
        self.overlay_intensity = overlay_intensity  # Overlay transparency
        self.dilation = dilation
        # Banana-specific post-processing parameters
        self.enable_banana_processing = enable_banana_processing # Master switch
        self.enable_shape_filtering = enable_shape_filtering
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_curvature = min_curvature
        self.max_curvature = max_curvature
        self.enable_ellipse_fitting = enable_ellipse_fitting
        
        # General parameters
        self.enable_crf = enable_crf
        self.simple_mode = False  # Start with simple mode by default
        
        # Initialize model with exact same configuration as training
        self.segm_model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes,
            #encoder_trainable=True,
            activation=None,  # Keep raw logits
            decoder_attention_type='scse'
        )
        
        # Load model weights
        state_dict = torch.load(model_path, map_location=self.device)
        #print("Model state dict keys:", state_dict.keys())
        
        # Load state dict with strict=True since we're matching the training config
        missing_keys, unexpected_keys = self.segm_model.load_state_dict(state_dict, strict=True)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        
        # Move model to appropriate device
        self.segm_model = self.segm_model.to(self.device)
        self.segm_model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        base_colors = [
                [0, 0, 0],    # background
                [0, 255, 0],  # class 1
                [0, 255, 255],  # class 2
                [255, 0, 0],  # class 3
                # Add more or generate on demand
            ]
        if self.num_classes <= len(base_colors):
            self.class_colors = {i: base_colors[i] for i in range(self.num_classes)}
        else:
            # Generate colors dynamically if needed
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('jet', self.num_classes)
            self.class_colors = {i: list((np.array(cmap(i)[:3]) * 255).astype(int)) for i in range(self.num_classes)}

    def set_banana_processing(self, enabled):
        """Master toggle for all banana-specific post-processing."""
        self.enable_banana_processing = enabled
        print(f"Banana-specific post-processing: {'ENABLED' if enabled else 'DISABLED'}")

    def set_confidence_threshold(self, threshold):
        """Update confidence threshold dynamically"""
        self.confidence_threshold = threshold
        print(f"Confidence threshold updated to: {threshold}")

    def set_temperature(self, temperature):
        """Update temperature scaling dynamically"""
        self.temperature = temperature
        print(f"Temperature updated to: {temperature}")

    def set_min_area(self, min_area):
        """Update minimum area threshold dynamically"""
        self.min_area = min_area
        print(f"Minimum area threshold updated to: {min_area}")

    def set_overlay_intensity(self, overlay_intensity):
        """Update overlay transparency dynamically"""
        self.overlay_intensity = overlay_intensity
        print(f"Overlay transparency updated to: {overlay_intensity}")

    def set_dilation(self, dilation):
        """Update overlay transparency dynamically"""
        self.dilation = dilation
        print(f"Dilation updated to: {dilation}")

    def set_shape_filtering(self, enabled):
        """Update shape filtering enable/disable"""
        self.enable_shape_filtering = enabled
        print(f"Shape filtering: {'enabled' if enabled else 'disabled'}")

    def set_crf(self, enabled):
        """Update spatial consistency enable/disable"""
        self.enable_crf = enabled
        print(f"Spatial consistency: {'enabled' if enabled else 'disabled'}")

    def set_aspect_ratio_range(self, min_ratio, max_ratio):
        """Set aspect ratio range for shape filtering."""
        self.min_aspect_ratio = min_ratio
        self.max_aspect_ratio = max_ratio
        print(f"Aspect ratio range: {min_ratio:.1f} - {max_ratio:.1f}")

    def set_curvature_range(self, min_curv, max_curv):
        """Set curvature range for shape filtering."""
        self.min_curvature = min_curv
        self.max_curvature = max_curv
        print(f"Curvature range: {min_curv:.1f} - {max_curv:.1f}")

    def set_simple_mode(self, enabled):
        """Update simple mode enable/disable"""
        self.simple_mode = enabled
        print(f"Simple mode: {'enabled' if enabled else 'disabled'}")

    def set_ellipse_fitting(self, enabled):
        """Update ellipse fitting enable/disable"""
        self.enable_ellipse_fitting = enabled
        print(f"Ellipse fitting: {'enabled' if enabled else 'disabled'}")

    def calculate_aspect_ratio(self, contour):
        """Calculate aspect ratio of a contour (width/height)"""
        x, y, w, h = cv2.boundingRect(contour)
        return w / h if h > 0 else 0

    def calculate_curvature(self, contour):
        """Calculate curvature of a contour using arc length to chord length ratio"""
        if len(contour) < 3:
            return 0
        
        # Calculate arc length
        arc_length = cv2.arcLength(contour, closed=True)
        
        # Calculate chord length (distance between start and end points)
        start_point = contour[0][0]
        end_point = contour[-1][0]
        chord_length = np.linalg.norm(end_point - start_point)
        
        # Curvature is arc_length / chord_length
        # Higher values indicate more curved shapes
        return arc_length / chord_length if chord_length > 0 else 1

    def filter_by_shape(self, binary_mask):
        if not self.enable_shape_filtering:
            return binary_mask
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary_mask
        
        filtered_mask = np.zeros_like(binary_mask)
        for contour in contours:
            aspect_ratio = self.calculate_aspect_ratio(contour)
            curvature = self.calculate_curvature(contour)
            area = cv2.contourArea(contour)
            
            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and \
               self.min_curvature <= curvature <= self.max_curvature and \
               area >= self.min_area:
                cv2.fillPoly(filtered_mask, [contour], 1)
        return filtered_mask

    def apply_morphological_filtering(self, mask):
        """Apply advanced morphological filtering for banana-like shapes"""
        # Create a banana-shaped kernel for morphological operations
        kernel_size = self.morph_kernel_size
        banana_kernel = np.zeros((kernel_size * 2 + 1, kernel_size * 2 + 1), np.uint8)
        
        # Create an elliptical kernel that approximates banana cross-section
        cv2.ellipse(banana_kernel, 
                   (kernel_size, kernel_size), 
                   (kernel_size, kernel_size // 2), 
                   0, 0, 360, 1, -1)
        
        # Apply morphological operations with banana-shaped kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, banana_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, banana_kernel)
        
        return mask

    def apply_spatial_consistency(self, mask):
        """Apply CRF-like spatial consistency to smooth boundaries"""
        if not self.enable_crf:
            return mask
        
        # Simple spatial consistency using Gaussian filtering and thresholding
        # This mimics CRF behavior without the computational complexity
        
        # Apply Gaussian smoothing
        smoothed = gaussian_filter(mask.astype(float), sigma=1.0)
        
        # Threshold to maintain sharp boundaries while smoothing
        threshold = 0.5
        smoothed_mask = (smoothed > threshold).astype(np.uint8)
        
        # Remove isolated pixels
        kernel = np.ones((3, 3), np.uint8)
        smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)
        
        return smoothed_mask

    def fit_banana_template(self, mask, class_id):
        """Fit a banana-shaped template to the predicted region"""
        class_mask = (mask == class_id).astype(np.uint8)
        
        # Find the largest contour
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit an ellipse to the contour
        if len(largest_contour) >= 5:  # Need at least 5 points for ellipse fitting
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                
                # Create a mask from the fitted ellipse
                template_mask = np.zeros_like(class_mask)
                cv2.ellipse(template_mask, ellipse, 1, -1)
                
                # Combine original mask with template (intersection)
                combined_mask = cv2.bitwise_and(class_mask, template_mask)
                
                # Fill holes in the combined mask
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                
                return combined_mask
            except:
                return class_mask
        
        return class_mask

    def fit_ellipse_to_mask(self, binary_mask):
        if not self.enable_ellipse_fitting:
            return binary_mask
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary_mask
        
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) < 5:
            return binary_mask
            
        try:
            ellipse = cv2.fitEllipse(largest_contour)
            ellipse_mask = np.zeros_like(binary_mask)
            cv2.ellipse(ellipse_mask, ellipse, 1, -1)
            return cv2.bitwise_and(binary_mask, ellipse_mask)
        except Exception:
            return binary_mask

    def perform_segmentation(self, frame):
        print("Input frame shape:", frame.shape)
        print(f"Current parameters - Confidence: {self.confidence_threshold}, Temperature: {self.temperature}")
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        print("Input tensor shape:", input_tensor.shape)

        with torch.no_grad():
            # Get model predictions
            logits = self.segm_model(input_tensor)
            
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            
            # Get probabilities
            probs = torch.softmax(scaled_logits, dim=1)
            
            # Get confidence scores
            confidence, predictions = torch.max(probs, dim=1)
            
            # Convert to numpy
            pred_mask = predictions.squeeze().cpu().numpy()
            conf_mask = confidence.squeeze().cpu().numpy()
            
            print("Raw output shape:", logits.shape)
            print("Confidence range:", conf_mask.min(), conf_mask.max())
            print("Unique values in output:", np.unique(pred_mask))
            print(f"Pixels per class before thresholding: {[np.sum(pred_mask == i) for i in range(3)]}")

            # Apply confidence threshold
            low_conf_mask = conf_mask < self.confidence_threshold
            pred_mask[low_conf_mask] = 0  # Set low confidence predictions to background
            
            print(f"Low confidence pixels: {np.sum(low_conf_mask)}")
            print(f"Pixels per class after thresholding: {[np.sum(pred_mask == i) for i in range(3)]}")

        return self.smooth_mask(pred_mask)

    def smooth_mask(self, mask):
        # Use simple mode by default for now
        if self.simple_mode:
            return self.smooth_mask_simple(mask)
        else:
            return self.smooth_mask_advanced(mask)

    def smooth_mask_advanced(self, mask):
        print("Advanced post-processing mode")
        smoothed_mask = np.zeros_like(mask)
        
        for class_id in np.unique(mask):
            if class_id == 0:
                continue
            
            class_mask = (mask == class_id).astype(np.uint8)
            print(f"Processing class {class_id}...")

            # Use banana-specific logic only for class 1 and only if master toggle is on
            if class_id == 1 and self.enable_banana_processing:
                print("  Applying OD-specific (banana) post-processing...")
                # Use a wide kernel to bridge gaps, controlled by the UI slider
                kernel_height = self.morph_kernel_size
                kernel_width = self.morph_kernel_size * 5  # Make kernel wide
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
                class_mask = binary_closing(class_mask, structure=kernel).astype(np.uint8)
                class_mask = binary_opening(class_mask, structure=kernel).astype(np.uint8)

                if self.enable_shape_filtering:
                    class_mask = self.filter_by_shape(class_mask)
                if self.enable_ellipse_fitting:
                    class_mask = self.fit_ellipse_to_mask(class_mask)
            else:  # Generic processing for other classes (e.g., spring)
                print("  Applying generic post-processing...")
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                class_mask = binary_opening(class_mask, structure=kernel).astype(np.uint8)
                class_mask = binary_closing(class_mask, structure=kernel).astype(np.uint8)

            if self.enable_crf:
                class_mask = self.apply_spatial_consistency(class_mask)

            labeled_mask = label(class_mask)
            regions = regionprops(labeled_mask)
            for region in regions:
                if region.area >= self.min_area:
                    smoothed_mask[labeled_mask == region.label] = class_id
        
        return smoothed_mask

    def smooth_mask_simple(self, mask):
        print("Simple post-processing mode")
        smoothed_mask = np.zeros_like(mask)
        
        for class_id in np.unique(mask):
            if class_id == 0:
                continue

            class_mask = (mask == class_id).astype(np.uint8)
            print(f"Processing class {class_id}...")

            # Use banana-specific logic only for class 1 and only if master toggle is on
            if class_id == 1 and self.enable_banana_processing:
                print("  Applying OD-specific (banana) gap closing...")
                kernel_height = self.morph_kernel_size
                kernel_width = self.morph_kernel_size * 5
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
                class_mask = binary_closing(class_mask, structure=kernel).astype(np.uint8)
                class_mask = binary_opening(class_mask, structure=kernel).astype(np.uint8)

                if self.enable_ellipse_fitting:
                    class_mask = self.fit_ellipse_to_mask(class_mask)
            else:  # Generic processing for other classes
                print("  Applying generic post-processing...")
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                class_mask = binary_opening(class_mask, structure=kernel).astype(np.uint8)
                class_mask = binary_closing(class_mask, structure=kernel).astype(np.uint8)

            labeled_mask = label(class_mask)
            regions = regionprops(labeled_mask)
            for region in regions:
                if region.area >= 50:
                    smoothed_mask[labeled_mask == region.label] = class_id
        
        return smoothed_mask


    # Then use self.class_colors in create_segmentation_overlay:
    def create_segmentation_overlay(self, frame, mask):
        unique_classes = np.unique(mask)
        print("Unique classes in mask:", unique_classes)
        assert np.all(np.isin(unique_classes, list(self.class_colors.keys()))), "Mask contains invalid class IDs."
        
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in self.class_colors.items():
            overlay[mask == class_id] = color

        # Resize overlay to match frame dimensions
        overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Blend overlay with the original frame
        return cv2.addWeighted(frame, 1 - self.overlay_intensity, overlay, self.overlay_intensity, 0)

