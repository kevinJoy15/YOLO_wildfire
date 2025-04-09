import torch
import torch.nn as nn
from ultralytics import YOLO
import re

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    
    CBAM sequentially applies channel attention and spatial attention to input features.
    This helps the model focus on important features both channel-wise and spatially.
    
    Reference: "CBAM: Convolutional Block Attention Module" - Woo et al., ECCV 2018
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        """
        Initialize CBAM module
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio for the MLP (default: 16)
            spatial_kernel: Kernel size for spatial attention (default: 7)
        """
        super(CBAM, self).__init__()
        
        # Channel Attention Module
        # Uses both average and max pooling followed by shared MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        
        # Shared MLP for channel attention
        self.shared_mlp = nn.Sequential(
            # Reduce channels by reduction factor
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # Restore original channel count
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()  # For attention weights

        # Spatial Attention Module
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=(spatial_kernel - 1) // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of convolutional layers using Kaiming normal"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass of CBAM
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor with same shape as input, but with attention applied
        """
        # 1. Channel Attention
        # Generate channel attention map using both avg and max pooling
        avg_out = self.shared_mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.shared_mlp(self.max_pool(x))  # [B, C, 1, 1]
        
        # Combine and apply sigmoid to get channel attention weights
        channel_att = self.sigmoid_channel(avg_out + max_out)  # [B, C, 1, 1]
        
        # Apply channel attention
        x_channel = x * channel_att  # Element-wise multiplication

        # 2. Spatial Attention
        # Generate spatial features by pooling across channels
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate avg and max pooled features
        spatial_features = torch.cat([avg_out_spatial, max_out_spatial], dim=1)  # [B, 2, H, W]
        
        # Generate and apply spatial attention map
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_features))  # [B, 1, H, W]
        
        # Apply spatial attention to channel-attended features
        return x_channel * spatial_att  # Final output with both attentions applied

def identify_model_sections(model):
    """
    Analyze the model structure to identify backbone, neck, and head sections
    based on architecture patterns.
    
    Args:
        model: YOLO model
        
    Returns:
        dict: Contains lists of indices for each section
    """
    
    # Print model structure to help with analysis
    print("Analyzing model structure...")
    
    # First identify the model structure
    for i, m in enumerate(model.model):
        module_name = type(m).__name__
        print(f"Layer {i}: {module_name}")


    backbone_range = range(0, 10)  # Typical backbone indices
    neck_range = range(10, 23)     # Typical neck indices
    # head_range = range(23, 27)     # Typical head indices don't really matter 
    
    return {
        "backbone": list(backbone_range),
        "neck": list(neck_range),
        # "head": list(head_range)
    }

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

def create_cbam_enhanced_yolo(model_path, apply_to, device=None):
    """
    Create a YOLO model with CBAM modules applied to specified sections.
    
    Args:
        model_path: Path to the YOLO model weights
        apply_to: Where to apply CBAM ('all', 'backbone', 'neck')
        device: Computing device to use
        
    Returns:
        Enhanced YOLO model
    """
    # Load the base YOLO model
    model = YOLO(model_path)
    
    print(f"\nEnhancing YOLO model with CBAM attention ({apply_to} strategy)")
    
    if apply_to == "all":                       # Apply to all Conv layers
        add_cbam_to_model(model.model)
    elif apply_to == "backbone":                # Apply only to backbone
        add_cbam_to_model(model.model, backbone_only=True)
    elif apply_to == "neck":                    # Apply only to neck
        add_cbam_to_model(model.model, neck_only=True)
    else:
        raise ValueError(f"Unknown strategy: {apply_to}. Choose from 'all', 'backbone', or 'neck'.")
    
    # Move model to the specified device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model.to(device)
    print(f"Model moved to device: {device}")
    print(f"YOLO model with CBAM attention ready")
    
    return model

# Example usage (can be commented out when importing this as a module)
if __name__ == "__main__":
    # Create a YOLO model with CBAM applied to the neck
    model = create_cbam_enhanced_yolo(
        model_path="yolo11n.pt",
        apply_to="neck",
        # device = "mps"
    )
    
    # model_architecture = str(model.model)
        
    # with open("architecture_checks/yolo11n_cbam.txt", "w") as f:
    #     f.write(model_architecture)
    
    print(model.model)
    
    # Train the model
    model.train(
        data="dataset/red_D-Fire/data.yaml", 
        epochs=100, 
        imgsz=512, 
        batch=16, 
        name="cbam_neck",
    )
    
    
    # # Load the model from the last checkpoint
    # model = create_cbam_enhanced_yolo(
    #     model_path="runs/detect/cbam_all/weights/last.pt",  # use the checkpoint file
    #     apply_to="all",
    #     device="mps"
    # )
    
    # # Optionally, you can pass a resume flag if the train() method supports it.
    # model.train(
    #     data="Dataset/new_fire&smoke/data.yaml", 
    #     epochs=50, 
    #     imgsz=512, 
    #     batch=16, 
    #     name="cbam_all",
    #     lr0=0.005,
    #     cos_lr=True,
    #     resume=True  # This tells the trainer to continue from the checkpoint's epoch
    # )
