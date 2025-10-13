import cv2
import numpy as np
from PIL import Image
import io

def resize_image(frame, target_size=(640, 640)):
    """
    Resize the frame to the target size.
    """
    return cv2.resize(frame, target_size)

def convert_to_rgb(frame):
    """
    Convert BGR frame to RGB for displaying.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def pil_image_from_array(frame):
    """
    Convert a NumPy array to a PIL Image.
    """
    return Image.fromarray(frame)

def image_to_bytes(img):
    """
    Convert a PIL Image to bytes for quick processing or saving.
    """
    byte_array = io.BytesIO()
    img.save(byte_array, format="PNG")
    return byte_array.getvalue()

def prepare_for_inference(frame):
    """
    Prepare image for YOLO inference, including resizing and normalizing if needed.
    """
    resized_frame = resize_image(frame)
    rgb_frame = convert_to_rgb(resized_frame)
    return rgb_frame
