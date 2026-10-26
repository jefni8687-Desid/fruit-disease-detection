import torch
import torch.nn as nn
import timm
class EfficientSwinNet(nn.Module):
    def __init__(
    self,
    num_classes=17,
    pretrained=False,
    dropout=0.4
    ):
        super().__init__()
 # ==================================================
 # EFFICIENTNET-B5
 # ==================================================
        self.efficientnet = timm.create_model(
    "efficientnet_b5",
    pretrained=pretrained,
    num_classes=0,
    global_pool="avg"
    )
    efficientnet_features = (
    self.efficientnet.num_features
    )
 # ==================================================
 # SWIN TRANSFORMER V2
 # ==================================================
    self.swin = timm.create_model(
    "swinv2_tiny_window8_256",
    pretrained=pretrained,
    num_classes=0,
    global_pool="avg"
    )
    swin_features = self.swin.num_features
 # ==================================================
 # FEATURE FUSION
    # ==================================================
    fusion_features = (
    efficientnet_features + swin_features
    )
    self.fusion = nn.Sequential(
    nn.Linear(
    fusion_features,
    1024
    ),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout)
    )
 # ==================================================
 # CLASSIFICATION HEAD
 # ==================================================
    self.classifier = nn.Sequential(
    nn.Linear(
    1024,
    512
    ),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(
    512,
    num_classes
    )
    )
    self.num_classes = num_classes
 # ======================================================
 # FORWARD PASS
 # ======================================================
def forward(self, x):
 # EfficientNet-B5 feature extraction
    efficient_features = self.efficientnet(x)
    # Swin Transformer V2 feature extraction
    swin_features = self.swin(x)
    # Feature fusion
    combined_features = torch.cat(
    [
    efficient_features,
    swin_features
    ],
    dim=1
    )
    # Fusion projection
    fused_features = self.fusion(
    combined_features
    )
    # Disease classification logits
    output = self.classifier(
    fused_features
    )
    return output
# ==========================================================
# MODEL CREATION FUNCTION
# ==========================================================
def create_model(
 num_classes=17,
 pretrained=False,
 dropout=0.4
):
    return EfficientSwinNet(
    num_classes=num_classes,
    pretrained=pretrained,
    dropout=dropout
    )
# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================
def load_model(
 model_path,
 num_classes=17,
 device=None
):
     if device is None:
        device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
        )
    else:
        device = torch.device(device)
        # Create the updated architecture
        model = create_model(
        num_classes=num_classes,
        pretrained=False
        )
 # Load checkpoint
        checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
        )
 # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
 # Remove DataParallel prefix
cleaned_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("module."):
        key = key[7:]
        cleaned_state_dict[key] = value
 # Load weights strictly
        model.load_state_dict(
        cleaned_state_dict,
        strict=True
        )
        model.to(device)
        model.eval()
    return model, device
# ==========================================================
# MODEL SUMMARY
# ==========================================================
def print_model_summary(model):
 total_params = sum(
 parameter.numel()
 for parameter in model.parameters()
 )
 trainable_params = sum(
 parameter.numel()
 for parameter in model.parameters()
 if parameter.requires_grad
 )
 print("=" * 55)
 print("EfficientSwinNet V2")
 print("Backbone 1 : EfficientNet-B5")
 print("Backbone 2 : Swin Transformer V2 Tiny")
 print(f"Classes : {model.num_classes}")
 print(f"Parameters : {total_params:,}")
 print(f"Trainable : {trainable_params:,}")
 print("=" * 55)
# ==========================================================
# TEST MODEL
# ==========================================================
if __name__ == "__main__":
 model = create_model(
 num_classes=17,
 pretrained=False
 )
 print_model_summary(model)
 # Both backbones expect 256 x 256 inputs
 sample_input = torch.randn(2, 3, 256, 256)
 model.eval()
 with torch.no_grad():
    output = model(sample_input)
    print("Input shape :", sample_input.shape)
    print("Output shape:", output.shape)
    assert output.shape == (2, 17)
    print("Model test passed!")
