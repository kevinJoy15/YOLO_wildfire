from ultralytics import YOLO
import CBAM_BiFPN
import torch

# def train_cbam_bifpn_yolo(data_yaml, model_weights="yolov8n.pt", epochs=100, imgsz=640):
#     # Load base YOLO model
#     model = YOLO(model_weights)
    
#     # Replace neck with CBAM-BiFPN (your implementation)
#     model = CBAM_BiFPN(model, num_bifpn_blocks=3)
    
#     # Training configuration
#     config = {
#         "data": data_yaml,
#         "epochs": epochs,
#         "imgsz": imgsz,
#         "batch": 16,
#         "device": "cuda" if torch.cuda.is_available() else "cpu",
#         "name": "cbam_bifpn_yolo",
#         "optimizer": "AdamW",  # Better than SGD for attention models
#         "lr0": 0.001,          # Initial learning rate
#         "weight_decay": 0.0005,
#     }
    
#     # Start training
#     results = model.train(**config)
#     return results

# if __name__ == "__main__":
#     train_cbam_bifpn_yolo(
#         data_yaml="dataset/data.yaml",
#         model_weights="yolov8n.pt",
#         epochs=100,
#         imgsz=640
#     )

model = YOLO("yolov11.yaml")

model.train(
    data="dataset/red_D-Fire/data.yaml",
    device='mps',
    epochs=100,
    imgsz=512,
    batch=16,
    name="vanilla",
    project="help_run",
)