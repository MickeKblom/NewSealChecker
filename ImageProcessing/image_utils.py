import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import os
import datetime
import json
import cv2
from shapely.geometry import Polygon
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch



def mask_to_bbox(mask_np: np.ndarray):
    """
    Convert a binary or multi-class mask to bounding box coordinates.
    
    Args:
        mask_np: numpy array mask (binary or multi-class)
    
    Returns:
        tuple (xmin, ymin, xmax, ymax) bounding box coordinates
    """
    pos = np.where(mask_np > 0)
    if pos[0].size == 0 or pos[1].size == 0:
        # No foreground found, return full image bbox
        return 0, 0, mask_np.shape[1], mask_np.shape[0]
    ymin, ymax = pos[0].min(), pos[0].max()
    xmin, xmax = pos[1].min(), pos[1].max()
    return xmin, ymin, xmax, ymax

def crop_image(image_np: np.ndarray, bbox: tuple):
    """
    Crop a numpy image array to the bounding box.
    
    Args:
        image_np: numpy array image
        bbox: tuple (xmin, ymin, xmax, ymax)
        
    Returns:
        Cropped numpy image
    """
    xmin, ymin, xmax, ymax = bbox
    return image_np[ymin:ymax+1, xmin:xmax+1]

def resize_image(image_np: np.ndarray, fixed_size: tuple):
    """
    Resize numpy array image using Pillow with bilinear interpolation.
    
    Args:
        image_np: numpy array image
        fixed_size: tuple (width, height)
    
    Returns:
        Resized numpy image array
    """
    img_pil = Image.fromarray(image_np)
    img_resized = img_pil.resize(fixed_size, Image.BILINEAR)
    return np.array(img_resized)

def generate_unique_image_id(quality_label="image"):
    # Generate timestamp for filename (no colon as it is invalid in Windows filenames)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_filename = f"{quality_label}_{timestamp}"
    return base_filename

def crop_and_resize(image_np: np.ndarray, bbox: tuple, fixed_size: tuple):
    """
    Crop a numpy image array to bbox and resize.
    
    Args:
        image_np: numpy array image
        bbox: tuple (xmin, ymin, xmax, ymax)
        fixed_size: (width, height)
    
    Returns:
        Resized cropped numpy image array
    """
    cropped = crop_image(image_np, bbox)
    resized = resize_image(cropped, fixed_size)
    return resized

def crop_and_resize_with_mask(image_np: np.ndarray, mask_np: np.ndarray, fixed_size: tuple):
    """
    Crop image and mask using mask bounding box and resize both.
    
    Args:
        image_np: numpy array image
        mask_np: numpy array binary or multi-class mask
        fixed_size: (width, height)
    
    Returns:
        Tuple of resized PIL Images: (image_resized, mask_resized)
    """
    bbox = mask_to_bbox(mask_np)
    
    cropped_img = crop_image(image_np, bbox)
    cropped_mask = crop_image(mask_np, bbox)
    
    img_pil = Image.fromarray(cropped_img)
    mask_pil = Image.fromarray((cropped_mask * 255).astype(np.uint8))
    
    img_resized = img_pil.resize(fixed_size, Image.BILINEAR)
    mask_resized = mask_pil.resize(fixed_size, Image.NEAREST)
    
    return img_resized, mask_resized


def convert_cv_qt(cv_img, display_width, display_height):
    rgb_image = cv_img[..., ::-1]  # Convert BGR to RGB
    rgb_image = np.ascontiguousarray(rgb_image)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_img = QImage(
        rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(qt_img).scaled(
        display_width, display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


def display_cropped_and_mask(cropped_img, cropped_mask):
    # Convert inputs to numpy arrays if needed
    if isinstance(cropped_img, Image.Image):
        cropped_img_np = np.array(cropped_img)
    elif isinstance(cropped_img, np.ndarray):
        cropped_img_np = cropped_img
    else:
        raise TypeError("cropped_img must be a numpy.ndarray or PIL.Image.Image")

    if isinstance(cropped_mask, Image.Image):
        cropped_mask_np = np.array(cropped_mask)
    elif isinstance(cropped_mask, np.ndarray):
        cropped_mask_np = cropped_mask
    else:
        raise TypeError("cropped_mask must be a numpy.ndarray or PIL.Image.Image")

    # Convert to uint8 if needed
    if cropped_img_np.dtype != np.uint8:
        cropped_img_np = (cropped_img_np * 255).astype(np.uint8) if cropped_img_np.max() <= 1 else cropped_img_np.astype(np.uint8)

    if cropped_mask_np.dtype != np.uint8:
        cropped_mask_np = (cropped_mask_np * 255).astype(np.uint8) if cropped_mask_np.max() <= 1 else cropped_mask_np.astype(np.uint8)

    # Convert mask to grayscale image
    cropped_mask_pil = Image.fromarray(cropped_mask_np).convert('L')
    cropped_img_pil = Image.fromarray(cropped_img_np)

    # Create a new blank image to stack vertically
    combined_width = max(cropped_img_pil.width, cropped_mask_pil.width)
    combined_height = cropped_img_pil.height + cropped_mask_pil.height
    combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

    # Paste the cropped image on top
    combined_img.paste(cropped_img_pil, (0, 0))

    # Paste the mask below the image
    # Convert mask to RGB so it matches combined_img mode for paste
    mask_rgb = cropped_mask_pil.convert('RGB')
    combined_img.paste(mask_rgb, (0, cropped_img_pil.height))

    # Show the combined image in one window
    combined_img.show(title="Cropped Image and Mask")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_yolo_predictions(image_np, detections, base_path, image_id):
    ensure_dir(os.path.join(base_path, "YOLO", "images"))
    ensure_dir(os.path.join(base_path, "YOLO", "labels"))
    
    # Save image
    image_path = os.path.join(base_path, "YOLO", "images", f"{image_id}.png")
    Image.fromarray(image_np).save(image_path)
    
    # Save detection labels in YOLOv8 txt format
    label_path = os.path.join(base_path, "YOLO", "labels", f"{image_id}.txt")
    with open(label_path, "w") as f:
        for det in detections:
            # det should include class_id, x_center, y_center, width, height normalized to [0, 1]
            # You might need to convert bounding boxes accordingly here
            line = " ".join(map(str, det))
            f.write(line + "\n")
"""        
def save_segmentation_predictions(image_np, mask_np, base_path, image_id):
    ensure_dir(os.path.join(base_path, "Segmentation", "images"))
    ensure_dir(os.path.join(base_path, "Segmentation", "labels"))
    
    Image.fromarray(image_np).save(os.path.join(base_path, "Segmentation", "images", f"{image_id}.png"))
    Image.fromarray(mask_np).save(os.path.join(base_path, "Segmentation", "labels", f"{image_id}.png"))
"""
# Your polygonization function (trimmed to essentials here)
def mask_to_polygons(mask, max_points=200, eps_frac=0.003, min_area_px=25, keep_holes=False):
    m = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    mode = cv2.RETR_TREE if keep_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(m, mode, cv2.CHAIN_APPROX_NONE)
    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        perim = cv2.arcLength(cnt, True)
        eps = max(1.0, eps_frac * perim)
        approx = cv2.approxPolyDP(cnt, eps, True)
        while len(approx) > max_points:
            eps *= 1.25
            approx = cv2.approxPolyDP(cnt, eps, True)
        poly = approx.reshape(-1, 2).astype(float)
        if len(poly) >= 3:
            polys.append(poly)
    return polys

# Initialize global COCO JSON structure once outside capture loop
coco_json = {
    "images": [],
    "annotations": [],
    "categories": [
        # Static example category - adapt to your classes
        {"id": 1, "name": "class_1", "supercategory": "none"},
        {"id": 2, "name": "class_2", "supercategory": "none"},
        # Add more as needed
    ],
}
# Counters for unique ids globally (or reset as needed)
image_counter = 0
annotation_counter = 0


def save_segmentation_predictions(image_np, mask_np, base_path, image_id):
    global coco_json, image_counter, annotation_counter

    # Paths
    imgs_dir = os.path.join(base_path, "Segmentation", "images")
    labels_dir = os.path.join(base_path, "Segmentation", "labels")
    json_path = os.path.join(base_path, "Segmentation", "annotations.json")
    ensure_dir(imgs_dir)
    ensure_dir(labels_dir)

    # Save image and mask as PNG
    Image.fromarray(image_np).save(os.path.join(imgs_dir, f"{image_id}.png"))
    mask_to_save = mask_np.astype(np.uint8)

    mask_to_save = (mask_np.astype(np.uint8)) * 255
    Image.fromarray(mask_to_save).save(os.path.join(labels_dir, f"{image_id}.png"))

    
    #if mask_to_save.ndim == 3 and mask_to_save.shape[2] == 1:
    #    mask_to_save = mask_to_save.squeeze(axis=2)
        
    #Image.fromarray(mask_to_save).save(os.path.join(labels_dir, f"{image_id}.png"))

    #Image.fromarray(mask_np).save(os.path.join(labels_dir, f"{image_id}.png"))

    # Add image metadata to COCO json (avoid duplicates)
    image_counter += 1
    image_info = {
        "id": image_counter,
        "file_name": f"{image_id}.png",
        "width": image_np.shape[1],
        "height": image_np.shape[0],
    }
    coco_json["images"].append(image_info)

    # For each class in the mask, generate polygons and annotations
    unique_classes = np.unique(mask_np)
    for class_id in unique_classes:
        if class_id == 0:
            continue  # Skip background

        class_mask = (mask_np == class_id).astype(np.uint8)
        polygons = mask_to_polygons(class_mask)

        for poly in polygons:
            segmentation = poly[:-1].ravel().tolist()  # flatten x,y coords except last point
            x_min, y_min = float(np.min(poly[:, 0])), float(np.min(poly[:, 1]))
            x_max, y_max = float(np.max(poly[:, 0])), float(np.max(poly[:, 1]))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            area = float(Polygon(poly).area)
            if len(segmentation) < 6 or area == 0:
                continue

            annotation_counter += 1
            ann = {
                "id": annotation_counter,
                "image_id": image_info["id"],
                "category_id": int(class_id),
                "segmentation": [segmentation],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
            coco_json["annotations"].append(ann)

    # Save/update the COCO-style annotations JSON file after each save
    with open(json_path, "w") as f:
        json.dump(coco_json, f, indent=2)
    print(f"Saved COCO annotations JSON with {len(coco_json['annotations'])} annotations.")

def save_cnn_cropped(cropped_img_np, cropped_mask_np, base_path, image_id):
    ensure_dir(os.path.join(base_path, "CNN", "cropped_images"))
    ensure_dir(os.path.join(base_path, "CNN", "cropped_masks"))
    ensure_dir(os.path.join(base_path, "CNN", "images"))
    
    # Save original cropped image
    Image.fromarray(cropped_img_np).save(os.path.join(base_path, "CNN", "cropped_images", f"{image_id}.png"))
    # Save cropped mask
    Image.fromarray(cropped_mask_np).save(os.path.join(base_path, "CNN", "cropped_masks", f"{image_id}.png"))


def smooth_mask(mask):
    smoothed_mask = np.zeros_like(mask)
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
            
        class_mask = (mask == class_id).astype(np.uint8)
        print(f"Processing class {class_id}...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        class_mask = binary_opening(class_mask, structure=kernel).astype(np.uint8)
        class_mask = binary_closing(class_mask, structure=kernel).astype(np.uint8)

        labeled_mask = label(class_mask)
        regions = regionprops(labeled_mask)
        for region in regions:
            if region.area >= 50:
                smoothed_mask[labeled_mask == region.label] = class_id
    
    return smoothed_mask


def apply_spatial_consistency(mask):
    """Apply CRF-like spatial consistency to smooth boundaries"""    
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




def dilate_mask_torch(mask01, pad_px):
    if pad_px <= 0:
        return mask01
    k = 2 * pad_px + 1
    kernel = torch.ones((1, 1, k, k), device=mask01.device)
    mask = mask01.unsqueeze(0).unsqueeze(0).float()  # shape: [1, 1, H, W]
    dilated = F.conv2d(mask, kernel, padding=pad_px)
    return (dilated > 0).float().squeeze(0).squeeze(0)

def morphological_opening(mask, kernel_size=3):
    eroded = erode_mask_torch(mask, kernel_size)
    opened = dilate_mask_torch(eroded, kernel_size // 2)
    return opened

def morphological_closing(mask, kernel_size=3):
    dilated = dilate_mask_torch(mask, kernel_size // 2)
    closed = erode_mask_torch(dilated, kernel_size)
    return closed

def erode_mask_torch(mask01, kernel_size):
    k = kernel_size
    kernel = torch.ones((1, 1, k, k), device=mask01.device)
    mask = mask01.unsqueeze(0).unsqueeze(0).float()
    eroded = F.conv2d(mask, kernel, padding=k//2)
    return (eroded == kernel.numel()).float().squeeze(0).squeeze(0)

def apply_spatial_consistency_torch(mask01, sigma=1.0, threshold=0.5):
    mask_img = mask01.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    blurred = TF.gaussian_blur(mask_img, kernel_size=5)
    smoothed_mask = (blurred.squeeze(0).squeeze(0) > threshold).float()
    return smoothed_mask

def mask_to_bbox_torch(mask_tensor):
    """Returns (xmin, ymin, xmax, ymax) from a binary mask tensor."""
    pos = torch.nonzero(mask_tensor > 0, as_tuple=False)
    if pos.size(0) == 0:
        return 0, 0, mask_tensor.shape[1], mask_tensor.shape[0]
    ymin = pos[:, 0].min().item()
    ymax = pos[:, 0].max().item()
    xmin = pos[:, 1].min().item()
    xmax = pos[:, 1].max().item()


def crop_tensor(tensor, bbox):
    xmin, ymin, xmax, ymax = bbox
    return tensor[:, ymin:ymax+1, xmin:xmax+1]  # assuming shape [C, H, W]

def resize_tensor(tensor: torch.Tensor, size, mode='nearest', align_corners=False):
    """
    Resizes [C,H,W] tensor to `size=(H,W)`.

    - Float tensors: use the provided mode.
    - Integer tensors (e.g., labels):
        * if mode == 'nearest': cast to float -> interpolate -> cast back
        * else: raise (non-nearest on labels is usually incorrect)
    """
    was_3d = (tensor.ndim == 3)
    if was_3d:
        tensor = tensor.unsqueeze(0)  # [1,C,H,W]

    is_float = tensor.dtype.is_floating_point
    if is_float:
        resized = F.interpolate(
            tensor, size=size, mode=mode,
            align_corners=(align_corners if mode in ('bilinear', 'bicubic') else None),
            antialias=(mode in ('bilinear', 'bicubic'))
        )
    else:
        if mode != 'nearest':
            raise ValueError(f"Resizing integer tensors with mode='{mode}' is not supported. Use 'nearest'.")
        resized = F.interpolate(tensor.float(), size=size, mode='nearest').to(tensor.dtype)

    if was_3d:
        resized = resized.squeeze(0)
    return resized

def generate_class_colors(num_classes):
    base_colors = [
        [0, 0, 0],
        [0, 255, 0],
        [0, 255, 255],
        [255, 0, 0],
    ]
    if num_classes <= len(base_colors):
        return {i: base_colors[i] for i in range(num_classes)}
    else:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('jet', num_classes)
        return {i: list((np.array(cmap(i)[:3]) * 255).astype(int)) for i in range(num_classes)}


def tensor_preprocess(img, size=(640, 640), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), src_bgr=True):
    # If numpy, convert to tensor
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)

    # If HWC, permute to CHW
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3]  # drop alpha if present
        img = img.permute(2, 0, 1)  # CHW

    # Convert to float [0,1]
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    else:
        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0

    # BGR → RGB if needed
    if src_bgr:
        img = img[[2, 1, 0], ...]

    # Resize
    img = img.unsqueeze(0)  # BCHW
    img = F.interpolate(img, size=size, mode='bilinear', align_corners=False)
    img = img.squeeze(0)

    # Normalize
    mean = torch.tensor(mean, device=img.device).view(3, 1, 1)
    std = torch.tensor(std, device=img.device).view(3, 1, 1)
    img = (img - mean) / std

    return img