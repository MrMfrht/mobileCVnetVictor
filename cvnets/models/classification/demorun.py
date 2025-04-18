import torch
import argparse
from cvnets.models.classification.mobilevit_v2 import MobileViTv2

# 1. Set up basic options
opts = argparse.Namespace()
setattr(opts, "model.classification.mitv2.width_multiplier", 1.0)
setattr(opts, "model.classification.n_classes", 1000)
setattr(opts, "model.layer.global_pool", "mean")
setattr(opts, "model.classification.mitv2.dropout", 0.0)
setattr(opts, "model.classification.mitv2.attn_dropout", 0.0)
setattr(opts, "model.classification.mitv2.ffn_dropout", 0.0)
setattr(opts, "model.classification.mitv2.attn_norm_layer", "layer_norm_2d")
setattr(opts, "common.enable_coreml_compatible_module", False)

# Add the missing attribute with a default value
setattr(opts, "model.classification.enable_layer_wise_lr_decay", False) # <-- Add this line

# 2. Instantiate the model
model = MobileViTv2(opts=opts)
model.eval()

# ... rest of your script ...

# 3. Create a dummy input tensor
dummy_input = torch.randn(1, 3, 256, 256)

# 4. Call the model instance with the input tensor
with torch.no_grad():
    feature_maps = model(dummy_input)

# 5. Process the feature maps
print(f"Obtained {len(feature_maps)} feature maps.")
for i, fm in enumerate(feature_maps):
    print(f"Feature map {i} shape: {fm.shape}")