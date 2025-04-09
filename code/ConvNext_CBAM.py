import torch
import torch.nn as nn
import timm
from ultralytics import YOLO
import re
import json

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
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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

# --- ConvNeXt BACKBONE WRAPPER ---
class ConvNeXtBackbone(nn.Module):
    def __init__(self, variant="convnext_tiny", pretrained=True, out_indices=(1, 2, 3)):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.out_channels = self.model.feature_info.channels()

    def forward(self, x):
        return self.model(x)

# --- REPLACE BACKBONE ---
def swap_yolo_backbone_with_convnext(model, convnext_variant="convnext_tiny"):
    convnext_backbone = ConvNeXtBackbone(variant=convnext_variant)

    if hasattr(model, "model") and isinstance(model.model, nn.Sequential):
        print("Replacing model.model[0:10] (YOLO backbone) with ConvNeXt")
        model.model = nn.Sequential(
            convnext_backbone,  # ConvNeXt returns feature list
            *list(model.model.children())[10:]  # Keep neck and head
        )
        return model

    raise AttributeError("Expected model.model to be nn.Sequential.")



# --- ADD CBAM ---
def identify_model_sections(model):
    backbone_range = range(0, 10)
    neck_range = range(10, 23)
    return {"backbone": list(backbone_range), "neck": list(neck_range)}

def add_cbam_to_model(model, backbone_only=False, neck_only=False, channel_threshold=64, default_reduction=16):
    """
    Recursively adds CBAM modules to suitable Conv2d layers in the model.
    
    Args:
        model (nn.Module): The YOLO model to modify.
        backbone_only (bool): If True, only wrap layers whose path contains backbone indices.
        neck_only (bool): If True, only wrap layers whose path contains neck indices.
        channel_threshold (int): Minimum number of output channels for a layer to be wrapped.
        default_reduction (int): Default reduction ratio for CBAM.
    
    Returns:
        nn.Module: The modified model.
    """
    # First identify model sections
    sections = identify_model_sections(model)
    backbone_indices = sections["backbone"]
    neck_indices = sections["neck"]
    
    print(f"Identified backbone indices: {backbone_indices}")
    print(f"Identified neck indices: {neck_indices}")
    
    # Helper function to decide if CBAM should be added.
    def should_add_cbam(path, module):
        if not isinstance(module, nn.Conv2d) or module.out_channels < channel_threshold:
            return False
            
        # Extract layer index if present in the path
        match = re.search(r'model\.(\d+)', path)
        
        if match:
            idx = int(match.group(1))
            
            # Apply section-specific constraints
            if backbone_only and idx not in backbone_indices:
                return False
                
            if neck_only and idx not in neck_indices:
                return False
                
        # If we can't determine the section from the path, use other identifiers
        elif (backbone_only or neck_only):
            # Fall back to string matching if needed
            in_backbone = any(f"backbone" in path or f".{i}." in path for i in backbone_indices)
            in_neck = any(f"neck" in path or f".{i}." in path for i in neck_indices)
            
            if backbone_only and not in_backbone:
                return False
                
            if neck_only and not in_neck:
                return False
                
        return True

    # Recursive function to traverse and wrap modules.
    added_count = [0]  # Use a mutable object to track count across recursion.
    
    def process_module(module, path=""):
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name
            if should_add_cbam(current_path, child):
                reduction = max(default_reduction, child.out_channels // 32)
                # Replace the Conv2d layer with a Sequential block: [original, CBAM].
                wrapped = nn.Sequential(child, CBAM(child.out_channels, reduction=reduction))
                setattr(module, name, wrapped)
                added_count[0] += 1
                print(f"Added CBAM to {current_path} (channels: {child.out_channels})")
            else:
                process_module(child, current_path)
    
    process_module(model)
    print(f"Total CBAM modules added: {added_count[0]}")
    return model

# --- CREATE MODEL ---
def create_cbam_or_convnext_yolo(model_path, apply_cbam_to="none", use_convnext=True, convnext_variant="convnext_tiny", device=None):
    model = YOLO(model_path)

    if use_convnext:
        model.model = swap_yolo_backbone_with_convnext(model.model, convnext_variant)

    if apply_cbam_to in ["all", "backbone", "neck"]:
        add_cbam_to_model(model.model, 
                          backbone_only=(apply_cbam_to == "backbone"), 
                          neck_only=(apply_cbam_to == "neck"))

    if device is None:
        device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    model = create_cbam_or_convnext_yolo(
        model_path="yolo11n.pt",
        apply_cbam_to="neck",  # options: "all", "backbone", "neck", "none"
        use_convnext=True,
        convnext_variant="convnext_tiny"
    )
    