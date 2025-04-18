import torch
import argparse
from cvnets.models.classification.mobilevit_v2 import MobileViTv2

# 1. Set up basic options (replace with actual options loading if needed)
# You might need to load these from a config file or command-line arguments
# using the project's argument parsing setup for a real use case.
opts = argparse.Namespace()
# Add any essential options required by MobileViTv2.__init__ or get_configuration
# For example, width_multiplier is used in get_configuration
setattr(opts, "model.classification.mitv2.width_multiplier", 1.0)
# Add other necessary default options if MobileViTv2 or its components require them
setattr(opts, "model.classification.n_classes", 1000) # Example default
setattr(opts, "model.layer.global_pool", "mean") # Example default
setattr(opts, "model.classification.mitv2.dropout", 0.0)
setattr(opts, "model.classification.mitv2.attn_dropout", 0.0)
setattr(opts, "model.classification.mitv2.ffn_dropout", 0.0)
setattr(opts, "model.classification.mitv2.attn_norm_layer", "layer_norm_2d")
setattr(opts, "common.enable_coreml_compatible_module", False) # Example default

# 2. Instantiate the model
# Ensure MobileViTv2 class has the modified forward method
model = MobileViTv2(opts=opts)
model.eval() # Set to evaluation mode if not training

# 3. Create a dummy input tensor
# Adjust batch size (1), channels (3), height (256), and width (256) as needed
# The configuration uses img_channels: 3
dummy_input = torch.randn(1, 3, 256, 256)

# 4. Call the model instance with the input tensor
# This internally calls the forward method
with torch.no_grad(): # Disable gradient calculation if only doing inference
    feature_maps = model(dummy_input)

# 5. Process the feature maps
# feature_maps is a list of tensors from different layers
print(f"Obtained {len(feature_maps)} feature maps.")
for i, fm in enumerate(feature_maps):
    print(f"Feature map {i} shape: {fm.shape}")

# You can now access individual feature maps, e.g., feature_maps[0], feature_maps[1], etc.