import torch
import torch.nn as nn
from ultralytics import YOLO
import re

class CBAM(nn.Module):
    """
    Improved Convolutional Block Attention Module (CBAM)
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        
        # Channel Attention Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Use a more conservative reduction ratio
        actual_reduction = min(reduction, channels // 8)
        actual_reduction = max(actual_reduction, 2)  # Ensure at least 2 channels
        
        # Shared MLP with more careful initialization
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // actual_reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // actual_reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention Module
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=(spatial_kernel - 1) // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
        
        # Smaller initial values to avoid disrupting pre-trained features
        self._initialize_weights(scale=0.1)
        
        # Trainable scaling factor for gradual integration
        self.attention_scale = nn.Parameter(torch.tensor(0.0))

    def _initialize_weights(self, scale=0.1):
        """Initialize weights with smaller values to avoid feature disruption"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down the initialized weights
                m.weight.data.mul_(scale)

    def forward(self, x):
        identity = x  # Store input for residual connection
        
        # Channel Attention
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x_channel = x * channel_att
        
        # Spatial Attention
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_features))
        
        # Apply spatial attention with gradual integration
        enhanced = x_channel * spatial_att
        
        # Residual connection with learnable scale to allow gradual integration
        # This helps avoid disrupting pre-trained features too much
        attention_scale = torch.sigmoid(self.attention_scale)
        result = (1 - attention_scale) * identity + attention_scale * enhanced
        
        return result

def identify_model_sections(model):
    """
    Analyze the model structure to identify backbone, neck, and head sections
    based on architecture patterns.
    """
    # More robust analysis of model structure
    try:
        # Try to access model elements to determine structure
        model_layers = model.model.model if hasattr(model.model, 'model') else model.model
        
        # Count the number of layers if possible
        try:
            num_layers = len(model_layers)
            print(f"Model has {num_layers} layers")
            
            # For YOLOv8, backbone typically ends around 1/3 of the way through
            # and neck continues until about 80-90% of the way through
            backbone_end = num_layers // 3
            neck_end = int(num_layers * 0.9)
            
            backbone_range = range(0, backbone_end)
            neck_range = range(backbone_end, neck_end)
            
            print(f"Dynamic backbone range: 0-{backbone_end-1}")
            print(f"Dynamic neck range: {backbone_end}-{neck_end-1}")
            
            return {
                "backbone": list(backbone_range),
                "neck": list(neck_range),
            }
        except (TypeError, AttributeError):
            # Fall back to default ranges if we can't determine dynamically
            print("Could not determine model structure dynamically, using default ranges")
    except Exception as e:
        print(f"Error analyzing model structure: {str(e)}")
        print("Using default section assignments")
    
    # Default values as fallback
    backbone_range = range(0, 10)  # Typical backbone indices
    neck_range = range(10, 23)     # Typical neck indices
    
    return {
        "backbone": list(backbone_range),
        "neck": list(neck_range),
    }

def add_cbam_to_model(model, backbone_only=False, neck_only=False, 
                     channel_threshold=64, default_reduction=16,
                     strategic_placement=True):
    """
    Adds CBAM modules to the model with improved placement strategy
    """
    sections = identify_model_sections(model)
    backbone_indices = sections["backbone"]
    neck_indices = sections["neck"]
    
    print(f"Identified backbone indices: {backbone_indices}")
    print(f"Identified neck indices: {neck_indices}")
    
    # Keep track of where we add CBAM for better control
    added_locations = []
    
    # Helper function with improved placement logic
    def should_add_cbam(path, module):
        if not isinstance(module, nn.Conv2d) or module.out_channels < channel_threshold:
            return False
            
        # Extract layer index if present in the path
        match = re.search(r'model\.(\d+)', path)
        
        if match:
            idx = int(match.group(1))
            
            # Strategic placement: add CBAM only at the end of blocks
            # This is more effective than adding after every Conv layer
            if strategic_placement:
                # Check if this is likely the last Conv in a block
                # This is a heuristic - adjust based on your model architecture
                next_idx = idx + 1
                is_block_end = False
                
                # Detect if this is the end of a block
                if next_idx >= len(model.model):
                    is_block_end = True
                else:
                    next_module = model.model[next_idx]
                    # If next is a different type (e.g. MaxPool, Upsample), this might be a block end
                    is_block_end = not isinstance(next_module, type(module))
                
                if not is_block_end:
                    return False
            
            # Apply section-specific constraints
            if backbone_only and idx not in backbone_indices:
                return False
                
            if neck_only and idx not in neck_indices:
                return False
                
        # If we can't determine the section from the path, use other identifiers
        elif (backbone_only or neck_only):
            in_backbone = any(f"backbone" in path or f".{i}." in path for i in backbone_indices)
            in_neck = any(f"neck" in path or f".{i}." in path for i in neck_indices)
            
            if backbone_only and not in_backbone:
                return False
                
            if neck_only and not in_neck:
                return False
                
        return True

    added_count = [0]
    
    def process_module(module, path=""):
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name
            
            if should_add_cbam(current_path, child):
                # Use a more conservative reduction ratio based on channel count
                out_channels = child.out_channels
                # Adaptive reduction ratio - less aggressive for fewer channels
                if out_channels <= 128:
                    reduction = 4  # Less aggressive for small channel counts
                elif out_channels <= 256:
                    reduction = 8
                else:
                    reduction = default_reduction
                
                # Replace with Sequential that preserves the original layer
                wrapped = nn.Sequential(
                    child,
                    CBAM(child.out_channels, reduction=reduction)
                )
                setattr(module, name, wrapped)
                added_count[0] += 1
                added_locations.append(current_path)
                print(f"Added CBAM to {current_path} (channels: {child.out_channels}, reduction: {reduction})")
            else:
                process_module(child, current_path)
    
    process_module(model)
    print(f"Total CBAM modules added: {added_count[0]}")
    print(f"Added at: {added_locations}")
    return model

def create_cbam_enhanced_yolo(model_path, apply_to, device=None, 
                             strategic_placement=True, freeze_backbone=False):
    """
    Create a YOLO model with CBAM modules applied to specified sections.
    
    Args:
        model_path: Path to the YOLO model weights
        apply_to: Where to apply CBAM ('all', 'backbone', 'neck')
        device: Computing device to use
        strategic_placement: Whether to use strategic placement at block ends
        freeze_backbone: Whether to freeze backbone weights during initial training
    """
    # Load the base YOLO model
    model = YOLO(model_path)
    
    print(f"\nEnhancing YOLO model with CBAM attention ({apply_to} strategy)")
    
    if apply_to == "all":
        add_cbam_to_model(model.model, strategic_placement=strategic_placement)
    elif apply_to == "backbone":
        add_cbam_to_model(model.model, backbone_only=True, strategic_placement=strategic_placement)
    elif apply_to == "neck":
        add_cbam_to_model(model.model, neck_only=True, strategic_placement=strategic_placement)
    else:
        raise ValueError(f"Unknown strategy: {apply_to}. Choose from 'all', 'backbone', or 'neck'.")
    
    # Freeze backbone layers if specified
    if freeze_backbone:
        print("Freezing backbone layers to stabilize training...")
        backbone_indices = identify_model_sections(model)["backbone"]
        try:
            # Try to access model.model first to check if it's a valid structure
            # Different YOLO versions may have different structures
            model_layers = model.model.model if hasattr(model.model, 'model') else model.model
            
            for idx in backbone_indices:
                try:
                    # Try to freeze by direct indexing
                    for param in model_layers[idx].parameters():
                        param.requires_grad = False
                    print(f"Froze backbone layer {idx}")
                except (IndexError, TypeError, AttributeError) as e:
                    print(f"Could not freeze layer {idx}: {str(e)}")
            
            print("Backbone frozen. Only neck, head, and attention modules will be trained initially.")
        except (AttributeError, TypeError) as e:
            print(f"Could not access model layers structure: {str(e)}")
            print("Failed to freeze backbone - will continue without freezing.")
    
    # Move model to the specified device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model.to(device)
    print(f"Model moved to device: {device}")
    print(f"YOLO model with CBAM attention ready")
    
    return model

# Example usage with improved training strategy
if __name__ == "__main__":
    try:
        # Phase 1: Train with frozen backbone for stability
        model = create_cbam_enhanced_yolo(
            model_path="yolo11n.pt",
            apply_to="neck",  # Start with attention only in the neck
            strategic_placement=True,  # Use strategic placement
            freeze_backbone=True  # Freeze backbone initially
        )
        
        # Training with more conservative parameters
        model.train(
            data="dataset/red_D-Fire/data.yaml", 
            epochs=50,  # Initial training phase
            imgsz=512, 
            batch=16,
            name="cbam_neck_phase1",
            lr0=0.001,  # Lower initial learning rate
            lrf=0.01,   # Final learning rate as a fraction of initial
            warmup_epochs=5,  # Longer warmup
            cos_lr=True,
        )
        
        # Phase 2: Unfreeze and train everything
        model = create_cbam_enhanced_yolo(
            model_path="runs/detect/cbam_neck_phase1/weights/last.pt",
            apply_to="neck", 
            strategic_placement=True,
            freeze_backbone=False  # Now unfreeze everything
        )
        
        # Continue training with adjusted parameters
        model.train(
            data="dataset/red_D-Fire/data.yaml", 
            epochs=50,  # Additional epochs
            imgsz=512, 
            batch=16, 
            name="cbam_neck_phase2",
            lr0=0.0005,  # Even lower learning rate for fine-tuning
            lrf=0.001,   # Final learning rate
            cos_lr=True,
        )
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        print("\nFallback to simpler training approach...")
        
        # Fallback to simpler approach without freezing
        model = create_cbam_enhanced_yolo(
            model_path="yolo11n.pt",
            apply_to="neck",
            strategic_placement=True,
            freeze_backbone=False  # No freezing in fallback mode
        )
        
        # Train with conservative parameters
        model.train(
            data="dataset/red_D-Fire/data.yaml", 
            epochs=100,
            imgsz=512, 
            batch=16, 
            name="cbam_neck_simple",
            lr0=0.0005,  # Start with low learning rate
            lrf=0.001,
            warmup_epochs=3,
            cos_lr=True,
        )