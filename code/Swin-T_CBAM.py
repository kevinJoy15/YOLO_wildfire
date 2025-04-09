import torch
import torch.nn as nn
import timm
import re
from ultralytics import YOLO

# --- CBAM MODULE ---
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=(spatial_kernel - 1) // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x_channel = x * channel_att
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_features))
        return x_channel * spatial_att

# --- SWIN BACKBONE WRAPPER ---
class SwinBackbone(nn.Module):
    def __init__(self, variant="swin_tiny_patch4_window7_224", pretrained=True, out_indices=(1, 2, 3)):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.out_channels = self.model.feature_info.channels()  # Ex: [192, 384, 768]

    def forward(self, x):
        return self.model(x)

# --- BACKBONE REPLACER ---
def swap_yolo_backbone_with_swin(model, swin_variant="swin_tiny_patch4_window7_224"):
    swin_backbone = SwinBackbone(variant=swin_variant)

    if hasattr(model, "model") and isinstance(model.model, nn.Sequential):
        print("Replacing model.model[0:10] (YOLO backbone) with Swin-T")
        model.model = nn.Sequential(
            swin_backbone,
            *list(model.model.children())[10:]
        )
        return model

    raise AttributeError("Expected model.model to be nn.Sequential.")

# --- CBAM ADDER ---
def identify_model_sections(model):
    return {"backbone": list(range(0, 10)), "neck": list(range(10, 23))}

def add_cbam_to_model(model, backbone_only=False, neck_only=False, channel_threshold=64, default_reduction=16):
    sections = identify_model_sections(model)
    backbone_indices = sections["backbone"]
    neck_indices = sections["neck"]

    print(f"Identified backbone indices: {backbone_indices}")
    print(f"Identified neck indices: {neck_indices}")

    def should_add_cbam(path, module):
        if not isinstance(module, nn.Conv2d) or module.out_channels < channel_threshold:
            return False
        match = re.search(r'model\.(\d+)', path)
        if match:
            idx = int(match.group(1))
            if backbone_only and idx not in backbone_indices:
                return False
            if neck_only and idx not in neck_indices:
                return False
        return True

    def process_module(module, path=""):
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name
            if should_add_cbam(current_path, child):
                reduction = max(default_reduction, child.out_channels // 32)
                setattr(module, name, nn.Sequential(child, CBAM(child.out_channels, reduction)))
                print(f"Added CBAM to {current_path} (channels: {child.out_channels})")
            else:
                process_module(child, current_path)

    process_module(model)
    return model

# --- CREATOR ---
def create_yolo_with_backbone(model_path, apply_cbam_to="none", use_swin=True, swin_variant="swin_tiny_patch4_window7_224", device=None):
    model = YOLO(model_path)

    if use_swin:
        model.model = swap_yolo_backbone_with_swin(model.model, swin_variant)

    if apply_cbam_to in ["all", "backbone", "neck"]:
        add_cbam_to_model(
            model.model,
            backbone_only=(apply_cbam_to == "backbone"),
            neck_only=(apply_cbam_to == "neck")
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

data_yaml_path = "/content/drive/MyDrive/Dissertation/dataset/red_D-Fire/data.yaml"

# --- USAGE ---
if __name__ == "__main__":
    data_yaml_path = "/path/to/your/data.yaml"
    
    model = create_yolo_with_backbone(
        model_path="yolo11n.yaml",  # <--- Use .yaml, not .pt
        apply_cbam_to="neck",       # Options: "all", "backbone", "neck", "none"
        use_swin=True,
        swin_variant="swin_tiny_patch4_window7_224"
    )

    model.train(
    data=data_yaml_path,
    device='cuda',
    epochs=200,
    imgsz=512,            # compromise between quality and memory
    batch=16,             # reduced to fit GPU
    name="v1_",
    project="/content/drive/MyDrive/Dissertation/runs/yolov11/red_D-Fire/200/CBAM_Swin",
    )