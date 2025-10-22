from Interfaces.segm_trt import SegmentationModelTRT


class SegmentationModel(SegmentationModelTRT):
    """Compatibility wrapper so existing calls SegmentationModel(model_path, ...) route to TRT."""
    def __init__(
        self,
        model_path,
        num_classes=2,
        confidence_threshold=0.3,
        temperature=1.5,
        min_area=100,
        src_bgr=True,
        device=None,
        input_size=(640, 640),
        return_to_input_size=False,
    ):
        super().__init__(
            onnx_path="Models/Segmentation/best.onnx",
            engine_path="Models/Segmentation/best.trt",
            input_size=input_size,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
            temperature=temperature,
            src_bgr=src_bgr,
            return_to_input_size=return_to_input_size,
            fp16=True,
        )