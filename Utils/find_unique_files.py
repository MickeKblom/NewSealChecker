# -*- coding: utf-8 -*-
"""
Find YOLO images that do NOT exist in the segmentation dataset (by normalized 'core id'),
and copy those images (and optionally their labels) to a 'unique' output dataset.

- Uniqueness is determined by your 'extract_superflex_core_id' normalization function.
- YOLO images folder is auto-inferred from the labels folder if not explicitly provided.

Author: M.
"""

import os
import re
import csv
import shutil
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# ----------------------------
# CONFIG — EDIT THESE PATHS
# ----------------------------
yolo_label_dir = r'D:\NewApplication\TestImgs\Yolo\train\labels'      # .txt label files live here
yolo_image_dir = None  # e.g. r'D:\SegC\Complete 3.v8i.yolov8\train\images' ; set to None to auto-infer from labels dir

segmentation_dir = r'D:\NewApplication\TestImgs\Segm\train'  # images + annotations file live here

output_dir = r'D:\NewApplication\TestImgs\Unique_from_YOLO_not_in_Seg'  # where to copy unique images (and labels)
copy_labels = True          # also copy their YOLO .txt labels
dry_run = False             # True = don't copy, just print what would happen

# Recognized image extensions
img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# ----------------------------
# HELPERS (your functions, plus a few small utilities)
# ----------------------------

def extract_superflex_core_id(filename: str) -> str:
    """
    Your normalization logic: remove .rf.* tail, extension, and a few repeated suffix patterns.
    """
    name = filename.lower()
    name = re.sub(r'\.rf\..*$', '', name)  # strip .rf.<random> tokens
    name = re.sub(r'\.(jpg|jpeg|png|bmp|tiff)$', '', name)
    for _ in range(5):
        name = re.sub(r'_(png|jpg|bmp|tiff|png_jpg|bmp_jpg|jpg_png)$', '', name)
    return name

def collect_core_ids(folder: str, exts: Tuple[str, ...]) -> Dict[str, List[str]]:
    """
    Map normalized core_id -> list of filenames in the given folder (non-recursive).
    Only files ending with the provided image extensions are considered.
    """
    core_map = defaultdict(list)
    for file in os.listdir(folder):
        if file.lower().endswith(exts):
            cid = extract_superflex_core_id(file)
            core_map[cid].append(file)
    return core_map

def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def infer_yolo_images_dir_from_labels(labels_dir: str) -> str:
    """
    If labels dir path ends with '\labels', try sibling '\images'.
    Otherwise, return the same parent + 'images' if present; else None.
    """
    base = os.path.normpath(labels_dir)
    parent = os.path.dirname(base)
    candidate = os.path.join(parent, 'images')
    return candidate if os.path.isdir(candidate) else None
def find_yolo_image_for_label(stem: str, image_dir: str) -> List[str]:
    """
    For a given label stem, find one or more matching image files in image_dir with allowed extensions.
    We don't assume only one match: if multiple versions exist, we return them all.
    """
    found = []
    try:
        for f in os.listdir(image_dir):
            f_lower = f.lower()
            if f_lower.startswith(stem.lower() + '.') and f_lower.endswith(img_exts):
                found.append(os.path.join(image_dir, f))
    except FileNotFoundError:
        pass
    return found

def collect_yolo_label_stems(labels_dir: str) -> List[str]:
    """
    Collect base names (stems) for all .txt files in labels_dir (non-recursive).
    """
    stems = []
    for f in os.listdir(labels_dir):
        if f.lower().endswith('.txt'):
            stem = os.path.splitext(f)[0]
            stems.append(stem)
    return stems

# ----------------------------
# MAIN LOGIC
# ----------------------------

def main():
    # 1) Validate / infer dirs
    if not os.path.isdir(yolo_label_dir):
        raise FileNotFoundError(f"YOLO labels directory not found: {yolo_label_dir}")

    images_dir = yolo_image_dir or infer_yolo_images_dir_from_labels(yolo_label_dir)
    if not images_dir or not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"YOLO images directory not found. Tried: {images_dir!r}. "
            f"Set 'yolo_image_dir' explicitly in the config."
        )

    if not os.path.isdir(segmentation_dir):
        raise FileNotFoundError(f"Segmentation directory not found: {segmentation_dir}")

    out_images = os.path.join(output_dir, 'images')
    out_labels = os.path.join(output_dir, 'labels')
    ensure_dir(out_images)
    if copy_labels:
        ensure_dir(out_labels)

    # 2) Build core-id set from segmentation images
    seg_core_map = collect_core_ids(segmentation_dir, img_exts)
    seg_core_ids: Set[str] = set(seg_core_map.keys())

    # 3) Read YOLO label stems and map to YOLO images + core-ids
    label_stems = collect_yolo_label_stems(yolo_label_dir)

    # Track which YOLO images belong to which normalized core-id
    yolo_core_to_images: Dict[str, List[str]] = defaultdict(list)
    yolo_core_to_label_paths: Dict[str, List[str]] = defaultdict(list)

    missing_images_for_labels = []

    for stem in label_stems:
        # Find matching YOLO image(s)
        imgs = find_yolo_image_for_label(stem, images_dir)

        # If image(s) found, compute core-id from those filenames; else fallback to label stem
        if imgs:
            for img_path in imgs:
                cid = extract_superflex_core_id(os.path.basename(img_path))
                yolo_core_to_images[cid].append(img_path)
                # record corresponding label path once per image (same stem)
                label_path = os.path.join(yolo_label_dir, f"{stem}.txt")
                if os.path.isfile(label_path):
                    yolo_core_to_label_paths[cid].append(label_path)
        else:
            # no image file found for this label; still consider the label's normalized stem
            cid = extract_superflex_core_id(stem)
            missing_images_for_labels.append(stem)
            # (We won't be able to copy an image, but it's useful to log.)

    # 4) Determine which YOLO core-ids are NOT in segmentation
    yolo_core_ids = set(yolo_core_to_images.keys()) | set(yolo_core_to_label_paths.keys())
    unique_core_ids = sorted(cid for cid in yolo_core_ids if cid not in seg_core_ids)

    # 5) Copy unique images (+ labels) and write manifest
    manifest_path = os.path.join(output_dir, 'manifest.csv')
    copied_image_count = 0
    copied_label_count = 0

    with open(manifest_path, 'w', newline='', encoding='utf-8') as mf:
        writer = csv.writer(mf)
        writer.writerow([
            'core_id', 'src_image_path', 'dst_image_path',
            'src_label_path', 'dst_label_path'
        ])

        for cid in unique_core_ids:
            image_paths = yolo_core_to_images.get(cid, [])
            label_paths = yolo_core_to_label_paths.get(cid, [])

            # Copy each image for this core-id
            for img_src in image_paths:
                dst_img = os.path.join(out_images, os.path.basename(img_src))
                if not dry_run:
                    shutil.copy2(img_src, dst_img)
                copied_image_count += 1
                # If copying labels, pick label(s) that correspond to the same stem as the image
                # (In usual YOLO structure there's one label per stem.)
                dst_label_written = ''
                src_label_used = ''
                if copy_labels and label_paths:
                    # Try to match by stem
                    img_stem = os.path.splitext(os.path.basename(img_src))[0].lower()
                    # find a label whose stem matches
                    matched = None
                    for lp in label_paths:
                        if os.path.splitext(os.path.basename(lp))[0].lower() == img_stem:
                            matched = lp
                            break
                    # fallback: use the first label if exact match not found
                    label_src = matched or label_paths[0]
                    dst_label = os.path.join(out_labels, os.path.basename(label_src))
                    if not dry_run:
                        shutil.copy2(label_src, dst_label)
                    copied_label_count += 1
                    src_label_used = label_src
                    dst_label_written = dst_label

                writer.writerow([
                    cid,
                    img_src,
                    dst_img,
                    src_label_used,
                    dst_label_written
                ])

    # 6) Summary
    print("\n==== SUMMARY ====")
    print(f"Segmentation images dir           : {segmentation_dir}")
    print(f"YOLO labels dir                   : {yolo_label_dir}")
    print(f"YOLO images dir                   : {images_dir}")
    print(f"Output dir                        : {output_dir}")
    print(f"Total segmentation core-ids       : {len(seg_core_ids)}")
    print(f"Total YOLO core-ids               : {len(yolo_core_ids)}")
    print(f"Unique core-ids (YOLO not in seg) : {len(unique_core_ids)}")
    print(f"Copied images                     : {copied_image_count}{' (dry run)' if dry_run else ''}")
    if copy_labels:
        print(f"Copied labels                     : {copied_label_count}{' (dry run)' if dry_run else ''}")
    if missing_images_for_labels:
        print(f"\nWARNING: {len(missing_images_for_labels)} label(s) had no matching image in YOLO images dir.")
        print("Examples:", ", ".join(missing_images_for_labels[:10]))
    print(f"\nManifest written to               : {manifest_path}")
    print("Done.")
    

if __name__ == '__main__':
    main()
