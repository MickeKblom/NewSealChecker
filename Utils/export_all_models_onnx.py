import torch
from ImageProcessing.inference import ImageInferencePipeline
from ultralytics import YOLO

def export_to_onnx(model, input_tensor, onnx_file_path, dynamic_axes=None):
    """
    Export single-input model to ONNX format.
    """
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_tensor,
            onnx_file_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes if dynamic_axes else {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    print(f"Model exported to {onnx_file_path}")

def export_yolo_onnx(yolo_weights_path):
    yolo_model = YOLO(yolo_weights_path)
    yolo_model.export(format="onnx")

def main():
    # Load models through your pipeline class which internally loads all models
    pipeline = ImageInferencePipeline(
        segm_model_path="Models/Segmentation/best.pth",
        cnn_model_path="Models/CNN/best.pth",
        yolo_model_path="Models/YOLO/best.pt"
    )


    # ----- Export CNN with 4-channel input (RGB + mask) -----
    cnn_input = torch.randn(1, 4, 512, 224)  # batch=1, 4 channels (RGB+mask), 512x224
    export_to_onnx(
        model=pipeline.classifier.model,  # Access underlying torch model inside your ODClassifierWrapper
        input_tensor=cnn_input,
        onnx_file_path="Models/CNN/best.onnx",
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
    )

    # ----- Export Segmentation Model -----
    segm_input = torch.randn(1, 3, 640, 640)  # batch=1, 3 channels RGB
    export_to_onnx(
        model=pipeline.segm_model.segm_model,  # Access underlying torch model inside your SegmentationModel wrapper
        input_tensor=segm_input,
        onnx_file_path="Models/Segmentation/best.onnx"
    )
    
    # ----- Export YOLO -----
    export_yolo_onnx("Models/YOLO/best.pt")

if __name__ == "__main__":
    main()
