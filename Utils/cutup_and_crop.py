import os
import re
import json
import math
import shutil
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


import numpy as np
from PIL import Image, ImageDraw, ImageFilter


## -*- coding: utf-8 -*-
"""
Cutout generator:
- Loads RGB images and YOLO labels from input folders.
- Loads OD segmentation mask from COCO (_annotations.coco.json).
- Computes ROI from OD mask's white region (bounding rectangle).
- Crops both image and mask to ROI, then letterboxes to 512x224 (black padding).
- Exports two variants for every image:
    1) output/ok/images & output/ok/masks           (no 'short' masking)
    2) output/shorts/images & output/shorts/masks   (with 'short' regions blacked out)
- All outputs are PNG.

Requirements:
    pip install pillow numpy pycocotools
"""

import os
import json
import math
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw

# If pycocotools is not available on Windows, try:
#   pip install pycocotools
# (Recent wheels exist; older advice was 'pycocotools-windows')
from pycocotools.coco import COCO

# -----------------------
# Configurations & class info
# -----------------------

nc = 5
class_names = ['OD', 'seal', 'short', 'spring', 'spring area2']

# INPUTS
INPUT_IMAGES_DIR = r'D:\NewApplication\TestImgs\UniqueS\train'
INPUT_LABELS_DIR = r'D:\NewApplication\TestImgs\Yolo\train\labels'
COCO_JSON_PATH   = r'D:\NewApplication\TestImgs\UniqueS\_annotations.coco.json'

# OUTPUT ROOTS
OUTPUT_DIR_SHORT = r'output/shorts'   # masked shorts variant
OUTPUT_DIR_OK    = r'output/ok'       # non-masked variant

# OUTPUT SIZE (width, height)
OUT_W, OUT_H = 512, 224

# Value used for white (foreground) in saved PNG masks
MASK_FOREGROUND_VALUE = 255  # stored as 0/255 in PNG

# Class name of the defect to "mask" in the shorts variant
SHORT_CLASS_NAME = 'short'

# Supported image file extensions
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images(folder: str) -> List[str]:
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]

def extract_superflex_core_id(filename):
    name = filename.lower()
    name = re.sub(r'\.rf\..*$', '', name)
    name = re.sub(r'\.(jpg|jpeg|png|bmp|tiff)$', '', name)
    for _ in range(5):
        name = re.sub(r'_(png|jpg|bmp|tiff|png_jpg|bmp_jpg|jpg_png)$', '', name)
    return name
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

def collect_core_ids(folder, exts):
    core_map = defaultdict(list)
    for file in os.listdir(folder):
        if file.lower().endswith(exts):
            cid = extract_superflex_core_id(file)
            core_map[cid].append(file)
    return core_map

def load_yolo_labels_for_image(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Loads YOLO txt labels for one image.
    Each line: <cls> <xc> <yc> <w> <h>  (all normalized [0,1])
    Returns list of tuples.
    """
    if not os.path.isfile(label_path):
        return []
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                labels.append((cls_id, xc, yc, w, h))
            except Exception:
                # Skip malformed lines
                continue
    return labels

def yolo_norm_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Convert normalized YOLO bbox to pixel xyxy.
    """
    cx = xc * img_w
    cy = yc * img_h
    bw = w * img_w
    bh = h * img_h
    x0 = int(round(cx - bw / 2.0))
    y0 = int(round(cy - bh / 2.0))
    x1 = int(round(cx + bw / 2.0))
    y1 = int(round(cy + bh / 2.0))
    # clip
    x0 = max(0, min(img_w - 1, x0))
    y0 = max(0, min(img_h - 1, y0))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    return x0, y0, x1, y1

def apply_blackout_rects(img: Image.Image, rects: List[Tuple[int,int,int,int]]) -> Image.Image:
    """
    Black out (fill with 0,0,0) the provided rectangles on a copy of the image.
    Rect format: (x0, y0, x1, y1), inclusive/exclusive doesn't matter for fill.
    """
    if not rects:
        return img.copy()
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for (x0, y0, x1, y1) in rects:
        # PIL rectangles are inclusive of the end pixel; to be safe, expand by 0
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    return out

def bbox_from_mask(mask_np: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Computes tight bounding box (x0, y0, x1, y1) of non-zero mask.
    Returns None if mask is empty.
    """
    ys, xs = np.where(mask_np > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1), int(y1)

def crop_and_letterbox(
    img: Image.Image,
    mask: Image.Image,
    roi_xyxy: Tuple[int,int,int,int],
    out_w: int,
    out_h: int
) -> Tuple[Image.Image, Image.Image]:
    """
    Crop both image and mask to ROI, then letterbox to (out_w, out_h) with black padding.
    - Image resized with bilinear; mask with nearest neighbor.
    - Mask saved as single-channel 'L'.
    """
    x0, y0, x1, y1 = roi_xyxy
    # PIL crop uses box (left, upper, right, lower) with right/low exclusive
    crop_box = (x0, y0, x1 + 1, y1 + 1)
    img_c = img.crop(crop_box)
    mask_c = mask.crop(crop_box)

    cw, ch = img_c.size
    # Compute scale to fit
    scale = min(out_w / cw, out_h / ch) if (cw > 0 and ch > 0) else 1.0
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    # Resize
    img_r = img_c.resize((new_w, new_h), resample=Image.BILINEAR)
    mask_r = mask_c.resize((new_w, new_h), resample=Image.NEAREST)

    # Letterbox
    canvas_img = Image.new('RGB', (out_w, out_h), color=(0, 0, 0))
    canvas_msk = Image.new('L', (out_w, out_h), color=0)
    left = (out_w - new_w) // 2
    top  = (out_h - new_h) // 2
    canvas_img.paste(img_r, (left, top))
    canvas_msk.paste(mask_r, (left, top))

    return canvas_img, canvas_msk

# ------------------------------------------------------------
# COCO helpers
# ------------------------------------------------------------

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

def crop_and_resize_with_mask(image_np: np.ndarray, mask_np: np.ndarray, fixed_size: Tuple[int,int]):
    """
    Crop image and mask using mask bounding box and resize both.
    """
    bbox = mask_to_bbox(mask_np)
    if bbox is None:
        raise RuntimeError("Mask is empty, no crop possible.")
    cropped_img = crop_image(image_np, bbox)
    cropped_mask = crop_image(mask_np, bbox)
    img_pil = Image.fromarray(cropped_img)
    # mask is binary 0/1 or 0/255; ensure in 0-255 range as uint8
    mask_pil = Image.fromarray((cropped_mask > 0).astype(np.uint8)*255)
    img_resized = img_pil.resize(fixed_size, Image.BILINEAR)
    mask_resized = mask_pil.resize(fixed_size, Image.NEAREST)
    return img_resized, mask_resized

def expand_bbox(bbox: Tuple[int,int,int,int], img_w:int, img_h:int, pad_frac:float):
    x0, y0, x1, y1 = bbox
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)
    ex0 = max(0, x0 - pad_x)
    ey0 = max(0, y0 - pad_y)
    ex1 = min(img_w - 1, x1 + pad_x)
    ey1 = min(img_h - 1, y1 + pad_y)
    return ex0, ey0, ex1, ey1

def apply_feathered_black_box(img: Image.Image, rect: Tuple[int, int, int, int], feather: int = 10) -> Image.Image:
    """
    Apply a feathered (blurred) black rectangle on the image.
    """
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(rect, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    black_img = Image.new('RGB', img.size, (0, 0, 0))
    return Image.composite(black_img, img, mask)

def apply_feathered_black_box_mask(mask: Image.Image, rect: Tuple[int,int,int,int], feather: int=10):
    """
    Apply feathered black box on mask, considering mask is single channel L mode.
    Black box sets region to zero, feathering smooth edge.
    """
    blank = Image.new('L', mask.size, 0)
    mask_box = Image.new('L', mask.size, 0)
    draw = ImageDraw.Draw(mask_box)
    draw.rectangle(rect, fill=255)
    mask_box = mask_box.filter(ImageFilter.GaussianBlur(radius=feather))
    # Composite the mask: wherever mask_box is white (feathered), set mask pixel closer to zero
    # Formula: new_mask = min(original, (1 - mask_box_alpha))
    # Since mask_box is 0-255, normalize
    mask_box_np = np.array(mask_box, dtype=np.float32) / 255.0
    mask_np = np.array(mask, dtype=np.float32) / 255.0
    new_mask_np = mask_np * (1.0 - mask_box_np)
    new_mask = Image.fromarray((new_mask_np*255).astype(np.uint8))
    return new_mask
def non_overlapping_random_box(roi, short_boxes, mask_np, W, H):
    import random
    max_attempts = 100
    x0_roi, y0_roi, x1_roi, y1_roi = roi
    roi_w = x1_roi - x0_roi + 1
    roi_h = y1_roi - y0_roi + 1

    for _ in range(max_attempts):
        w = random.randint(20, 130)
        h = random.randint(20, 70)
        if w > roi_w or h > roi_h:
            continue
        x0 = random.randint(x0_roi, x1_roi - w)
        y0 = random.randint(y0_roi, y1_roi - h)
        candidate = (x0, y0, x0 + w, y0 + h)

        # Check for sufficient overlap with OD mask (at least 25%)
        mask_crop = mask_np[y0:y0+h, x0:x0+w]
        overlap_ratio = np.sum(mask_crop > 0) / (w*h)
        if overlap_ratio < 0.25:
            continue

        # Check no overlap with any padded short box
        overlap_with_short = False
        for sb in short_boxes:
            ix0 = max(candidate[0], sb[0])
            iy0 = max(candidate[1], sb[1])
            ix1 = min(candidate[2], sb[2])
            iy1 = min(candidate[3], sb[3])
            if ix1 > ix0 and iy1 > iy0:
                overlap_area = (ix1 - ix0) * (iy1 - iy0)
                if overlap_area > 0:
                    overlap_with_short = True
                    break
        if overlap_with_short:
            continue

        # Found candidate
        return candidate
    return None

def build_coco_indices(coco_json_path: str) -> Tuple[COCO, Dict[str,int], int]:
    """
    Loads COCO and builds:
      - file_name -> image_id map
      - od_cat_id for category with name 'OD' (case-insensitive)
    """
    coco = COCO(coco_json_path)
    # Build file_name->id map
    file_to_imgid = {}
    for img_id, img_info in coco.imgs.items():
        file_to_imgid[img_info['file_name']] = img_id

    # Find OD category id
    od_cat_id = None
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    for c in cats:
        if str(c.get('name', '')).strip().lower() == 'od':
            od_cat_id = c['id']
            break
    if od_cat_id is None:
        raise RuntimeError("Could not find category named 'OD' in COCO JSON.")
    return coco, file_to_imgid, od_cat_id

def build_od_mask(coco: COCO, img_id: int, od_cat_id: int, expected_size: Tuple[int,int]) -> Optional[Image.Image]:
    """
    Builds a binary OD mask (L, 0/255) for the given img_id from COCO.
    Returns None if no OD annotation is present.
    """
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[od_cat_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    if not anns:
        return None

    # Merge all OD instances (though you stated single OD per image)
    masks = []
    for ann in anns:
        m = coco.annToMask(ann)  # HxW uint8 (0/1)
        masks.append(m.astype(np.uint8))

    msum = np.sum(np.stack(masks, axis=0), axis=0)
    bin_mask = (msum > 0).astype(np.uint8) * MASK_FOREGROUND_VALUE

    h, w = bin_mask.shape
    exp_w, exp_h = expected_size
    if (w, h) != (exp_w, exp_h):
        # Safety: if sizes don't match, resize mask to image size (nearest)
        bin_mask_img = Image.fromarray(bin_mask, mode='L').resize((exp_w, exp_h), resample=Image.NEAREST)
    else:
        bin_mask_img = Image.fromarray(bin_mask, mode='L')

    return bin_mask_img

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def main():
    # Resolve class id for 'short'
    try:
        short_cls_id = class_names.index(SHORT_CLASS_NAME)
    except ValueError:
        raise RuntimeError(f"Class '{SHORT_CLASS_NAME}' not found in class_names {class_names}.")

    # Prepare outputs
    out_ok_img = os.path.join(OUTPUT_DIR_OK, 'images')
    out_ok_msk = os.path.join(OUTPUT_DIR_OK, 'masks')
    out_sh_img = os.path.join(OUTPUT_DIR_SHORT, 'images')
    out_sh_msk = os.path.join(OUTPUT_DIR_SHORT, 'masks')
    for d in [out_ok_img, out_ok_msk, out_sh_img, out_sh_msk]:
        ensure_dir(d)

    # Load COCO and indices
    print("Loading COCO annotations...")
    coco, file_to_imgid, od_cat_id = build_coco_indices(COCO_JSON_PATH)
    
    # Collect core IDs for images and labels using your functions
    image_core_map = collect_core_ids(INPUT_IMAGES_DIR, IMG_EXTS)
    label_core_map = collect_core_ids(INPUT_LABELS_DIR, ('.txt',))

    # Find core IDs present in both images and labels
    matching_core_ids = set(image_core_map.keys()) & set(label_core_map.keys())

    processed = 0
    skipped_no_od = 0
    skipped_not_in_coco = 0

    for core_id in matching_core_ids:
        # Pick one image and label file per core_id
        image_file = image_core_map[core_id][0]
        label_file = label_core_map[core_id][0]

        img_path = os.path.join(INPUT_IMAGES_DIR, image_file)
        label_path = os.path.join(INPUT_LABELS_DIR, label_file)

        # Find img_id in COCO for this image file
        # Try exact match
        if image_file in file_to_imgid:
            img_id = file_to_imgid[image_file]
        else:
            # Try basename match
            alt_ids = [img_id for (fn, img_id) in file_to_imgid.items() if os.path.basename(fn) == image_file]
            if alt_ids:
                img_id = alt_ids[0]
            else:
                skipped_not_in_coco += 1
                continue

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Cannot open image {img_path}: {e}")
            continue
        W, H = img.size

        od_mask_img = build_od_mask(coco, img_id, od_cat_id, expected_size=(W, H))
        if od_mask_img is None:
            skipped_no_od += 1
            continue

        mask_np = np.array(od_mask_img, dtype=np.uint8)
        roi = bbox_from_mask(mask_np)
        if roi is None:
            skipped_no_od += 1
            continue
        x0, y0, x1, y1 = roi

        yolo_labels = load_yolo_labels_for_image(label_path)

        # Skip if any short bbox > 110x60 px
        skip_image = False
        for (cls_id, xc, yc, bw, bh) in yolo_labels:
            if cls_id == short_cls_id:
                bw_px = int(round(bw * W))
                bh_px = int(round(bh * H))
                if bw_px > 110 or bh_px > 60:
                    skip_image = True
                    break
        if skip_image:
            continue

        # Expand short boxes by 10% padding
        short_rects_expanded = []
        for (cls_id, xc, yc, bw, bh) in yolo_labels:
            if cls_id == short_cls_id:
                rect = yolo_norm_to_xyxy(xc, yc, bw, bh, W, H)
                padded_rect = expand_bbox(rect, W, H, 0.1)
                short_rects_expanded.append(padded_rect)

        # --- OK variant ---
        img_ok_variant = img.copy()
        mask_ok_variant = od_mask_img.copy()
        for rect in short_rects_expanded:
            img_ok_variant = apply_feathered_black_box(img_ok_variant, rect, feather=10)
            mask_ok_variant = apply_feathered_black_box_mask(mask_ok_variant, rect, feather=10)

        # --- Shorts variant ---
        rand_rect = non_overlapping_random_box(roi, short_rects_expanded, mask_np, W, H)

        if rand_rect is None:
            img_short_variant = img.copy()
            mask_short_variant = od_mask_img.copy()
        else:
            img_short_variant = apply_feathered_black_box(img.copy(), rand_rect, feather=10)
            mask_short_variant = apply_feathered_black_box_mask(od_mask_img.copy(), rand_rect, feather=10)

        # Crop and resize both variants and masks
        img_ok_out, mask_ok_out = crop_and_resize_with_mask(np.array(img_ok_variant), np.array(mask_ok_variant), (OUT_W, OUT_H))
        img_sh_out, mask_sh_out = crop_and_resize_with_mask(np.array(img_short_variant), np.array(mask_short_variant), (OUT_W, OUT_H))

        # Save outputs
        stem, _ = os.path.splitext(image_file)
        ok_img_out_path = os.path.join(out_ok_img, f"{stem}.png")
        ok_msk_out_path = os.path.join(out_ok_msk, f"{stem}.png")
        sh_img_out_path = os.path.join(out_sh_img, f"{stem}.png")
        sh_msk_out_path = os.path.join(out_sh_msk, f"{stem}.png")

        img_ok_out.save(ok_img_out_path, format='PNG')
        mask_ok_out.save(ok_msk_out_path, format='PNG')
        img_sh_out.save(sh_img_out_path, format='PNG')
        mask_sh_out.save(sh_msk_out_path, format='PNG')

        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} images...")

    print("\n===== SUMMARY =====")
    print(f"Processed images         : {processed}")
    print(f"Skipped (no OD mask)     : {skipped_no_od}")
    print(f"Skipped (not in COCO)    : {skipped_not_in_coco}")
    print(f"Output (OK)    images -> {out_ok_img}")
    print(f"Output (OK)    masks  -> {out_ok_msk}")
    print(f"Output (SHORT) images -> {out_sh_img}")
    print(f"Output (SHORT) masks  -> {out_sh_msk}")
    print("Done.")

if __name__ == "__main__":
    main()