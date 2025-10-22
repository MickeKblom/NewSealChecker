from ultralytics import YOLO
import numpy as np
import cv2
import torch

class YOLOModel:
    def __init__(self, weights_path='best.pt', device: str | torch.device | None = None):
        self.device = torch.device(device) if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.model = YOLO(weights_path)

    @torch.inference_mode()
    def perform_detection(self, frame, confidence_level=0.25, classes_to_detect=None):
        if isinstance(frame, torch.Tensor):
            if frame.ndim == 3 and frame.shape[0] in (3, 4):
                frame = frame[:3, ...].permute(1, 2, 0).contiguous().to('cpu').numpy()
            elif frame.ndim == 3 and frame.shape[-1] in (3, 4):
                frame = frame[..., :3].to('cpu').numpy()
            else:
                raise ValueError("Unsupported tensor shape for YOLO input")

        results = self.model.predict(
            frame, imgsz=640, conf=confidence_level, classes=classes_to_detect,
            device=str(self.device), verbose=False
        )
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names,
        }

    def create_detections_overlay(self, frame, detections):
        boxes = detections.get('boxes', np.zeros((0, 4), dtype=float))
        scores = detections.get('scores', np.zeros((0,), dtype=float))
        labels = detections.get('labels', np.zeros((0,), dtype=float))
        class_names = detections.get('class_names', {})
        COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(label)
            class_name = class_names.get(class_id, str(class_id))
            color = COLORS[class_id % len(COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame





