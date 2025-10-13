from Interfaces.segm import SegmentationModel
from Interfaces.ODPredict import ODClassifierWrapper
from Interfaces.yolo import YOLOModel  
from ImageProcessing.image_utils import crop_and_resize_with_mask, morphological_opening, apply_spatial_consistency_torch, dilate_mask_torch, mask_to_bbox_torch, crop_tensor, resize_tensor
from device_config import DeviceConfig, Backend
import torch

class ImageInferencePipeline:
    def __init__(self, segm_model_path, cnn_model_path, yolo_model_path, device_cfg: DeviceConfig):
        self.segm_model = SegmentationModel(segm_model_path)
        self.classifier = ODClassifierWrapper(cnn_model_path)
        self.yolo_model = YOLOModel(yolo_model_path)
        self.device_cfg = device_cfg
        self.backend = device_cfg.backend

        if self.backend == Backend.TORCH:
                self.torch = torch
                self._init_torch(segm_model_path, cnn_model_path, yolo_model_path, device_cfg)
        elif self.backend == Backend.ORT:
            self._init_onnxruntime(segm_model_path, cnn_model_path, yolo_model_path, device_cfg)
        elif self.backend == Backend.TRT:
            self._init_tensorrt(segm_model_path, cnn_model_path, yolo_model_path, device_cfg)
        else:
            raise ValueError("Unknown backend")

    def _init_torch(self, segm_path, cnn_path, yolo_path, cfg: DeviceConfig):
        dev = self.torch.device(cfg.device)
        # Move models to device if they are torch modules
        # self.segm_model.to(dev).eval()
        self.torch_device = dev

    def _init_onnxruntime(self, segm_path, cnn_path, yolo_path, cfg: DeviceConfig):
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        providers = [cfg.device] if cfg.device else ort.get_available_providers()
        self.ort = {"providers": providers}

    def _init_tensorrt(self, segm_engine, cnn_engine, yolo_engine, cfg: DeviceConfig):
        self.trt = {"engines": (segm_engine, cnn_engine, yolo_engine)}

    def process_image(self, image_np, predict_segm=False, predict_class=False, predict_yolo=False, postprocess_enabled=False,
            dilate_kernel=5, dilate_iters=1, smooth_alpha=None):

        results = {
            'segmentation_mask': None,
            'class_prediction': None,
            'yolo_detections': None,
            'cropped_img': None,
            'cropped_mask': None,
        }
        print(self.backend)
        if self.backend == Backend.TORCH:
            img_t = torch.from_numpy(image_np).to(self.torch_device)
            # Ensure CHW for downstream tensor utilities
            if img_t.ndim == 3 and img_t.shape[-1] in (3, 4):
                img_chw = img_t.permute(2, 0, 1).contiguous()
            else:
                img_chw = img_t

            if predict_segm:
                # Segmentation model can accept HWC or CHW; returns [H,W]
                mask = self.segm_model.perform_segmentation(img_t)
                mask = mask.to(self.torch_device)

                if postprocess_enabled:
                    mask = morphological_opening(mask, kernel_size=3)
                    mask = apply_spatial_consistency_torch(mask, threshold=0.5)
                    mask = dilate_mask_torch(mask, pad_px=5)

                results['segmentation_mask'] = mask

                if predict_class:
                    # Crop using CHW image and [H,W] mask bbox
                    bbox = mask_to_bbox_torch(mask)
                    cropped_img = crop_tensor(img_chw, bbox)  # [C,H,W]
                    cropped_mask = crop_tensor(mask.unsqueeze(0), bbox).squeeze(0)  # [H,W]

                    # Convert to float prior to bilinear resize
                    img_for_resize = cropped_img.float() / 255.0 if not cropped_img.dtype.is_floating_point else cropped_img
                    # Resize to (H,W) = (224,512) as trained: 512x224
                    resized_img = resize_tensor(img_for_resize, size=(224, 512), mode='bilinear', align_corners=False)
                    resized_mask = resize_tensor(cropped_mask.unsqueeze(0), size=(224, 512), mode='nearest').squeeze(0)

                    pred_class = self.classifier.predict(resized_img, resized_mask)

                    results['class_prediction'] = pred_class
                    results['cropped_img'] = cropped_img
                    results['cropped_mask'] = cropped_mask
                    # Keep a copy of the actual CNN inputs (for optional debug visualization)
                    results['cnn_input_img'] = resized_img  # [C,224,512] float
                    results['cnn_input_mask'] = resized_mask  # [224,512]

            if predict_yolo:
                detections = self.yolo_model.perform_detection(img_t)
                results['yolo_detections'] = detections

        else:
            raise NotImplementedError("process_image only supports TORCH backend currently.")

        return results