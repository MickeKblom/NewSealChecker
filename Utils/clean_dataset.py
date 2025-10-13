import os
import json
import shutil

def clean_coco_dataset(base_path, annotation_filename, label_to_remove="OD"):
    train_path = os.path.join(base_path, "train")
    cleaned_train_path = os.path.join(base_path, "train_cleaned")
    os.makedirs(cleaned_train_path, exist_ok=True)

    annotation_path = os.path.join(train_path, annotation_filename)

    # Load COCO annotation file
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Map category id to category name for quick lookup
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Find all image IDs that have at least one annotation with the label_to_remove
    image_ids_to_remove = set()
    for ann in coco_data['annotations']:
        cat_name = categories.get(ann['category_id'], None)
        if cat_name == label_to_remove:
            image_ids_to_remove.add(ann['image_id'])

    # Filter out images and annotations belonging to images_to_remove
    images_to_keep = [img for img in coco_data['images'] if img['id'] not in image_ids_to_remove]
    annotations_to_keep = [ann for ann in coco_data['annotations'] if ann['image_id'] not in image_ids_to_remove]

    # Copy images to cleaned_train_path folder
    for img in images_to_keep:
        src_img_path = os.path.join(train_path, img['file_name'])
        dst_img_path = os.path.join(cleaned_train_path, img['file_name'])
        if os.path.isfile(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image file {src_img_path} not found!")

    # Update the coco_data with cleaned images and annotations
    coco_data['images'] = images_to_keep
    coco_data['annotations'] = annotations_to_keep

    # Save cleaned COCO JSON annotation in the cleaned folder
    cleaned_annotation_path = os.path.join(cleaned_train_path, annotation_filename.replace(".json", "_cleaned.json"))
    with open(cleaned_annotation_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Cleaned dataset created at: {cleaned_train_path}")
    print(f"Cleaned annotations saved to: {cleaned_annotation_path}")
    print(f"Removed {len(image_ids_to_remove)} images labeled '{label_to_remove}'.")

if __name__ == "__main__":
    base_path = r"D:\NewApplication\Utils\CleanedSegm 3.v2i.coco-segmentation"
    annotation_file = "_annotations.coco.json"
    clean_coco_dataset(base_path, annotation_file, label_to_remove="OD")
