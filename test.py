from ultralytics import YOLO

# od_model = YOLO("yolov8x.pt").to("cuda")

# od_model.export(format="engine", half=True, dynamic=True, simplify=True)

print("Modified TensorRT model")

tensorrt_model_new = YOLO("/home/ubuntu/Downloads/airport/yolov8x.engine")

od_preds = tensorrt_model_new.predict(
    "https://ultralytics.com/images/bus.jpg", device="cuda"
)

print("Original PyTorch model")

pytorch_model = YOLO("/home/ubuntu/Downloads/airport/yolov8x.pt")

od_preds = pytorch_model.predict(
    "https://ultralytics.com/images/bus.jpg", device="cuda"
)
