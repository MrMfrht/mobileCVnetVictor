import torch
from cvnets.models.classification.mobilevit_v2 import MobileViTv2
# Import necessary components for argument parsing
import argparse
from options.opts import get_training_arguments # Assuming this function aggregates most common/training args

# 1. Create the main argument parser
parser = argparse.ArgumentParser(description="MobileViTv2 Feature Extraction Demo")

# 2. Add arguments defined in the repository
# Add general training arguments (which include many defaults needed by layers/models)
parser = get_training_arguments(parser)

# Add model-specific arguments for MobileViTv2
parser = MobileViTv2.add_arguments(parser)

# Add any other argument groups if needed by base classes or specific layers,
# although get_training_arguments likely covers most common ones.
# For example:
# from cvnets.layers import ConvLayer2d
# parser = ConvLayer2d.add_arguments(parser) # If ConvLayer2d defines specific args

# 3. Parse arguments with an empty list to get defaults
# Provide an empty list to parse_args to use default values
opts = parser.parse_args([])

# --- Optional: Set specific options if defaults aren't quite right ---
# For example, if the default width multiplier needs changing for a specific variant:
# opts.model.classification.mitv2.width_multiplier = 0.75
# Or ensure necessary keys exist if not covered by defaults (less likely now)
# if not hasattr(opts.model.classification, 'enable_layer_wise_lr_decay'):
#     opts.model.classification.enable_layer_wise_lr_decay = False # Example fallback

# 4. Instantiate the model using the parsed opts object
model = MobileViTv2(opts=opts)
model.eval() # Set the model to evaluation mode

# Create a dummy input image
img = torch.randn(1, 3, 256, 256) # Assuming default input size, adjust if needed

# Pass the image through the model to get feature maps
feature_maps = model(img)

# Print the shape of each feature map
print(f"Received {len(feature_maps)} feature maps.")
for i, feat_map in enumerate(feature_maps):
    if isinstance(feat_map, torch.Tensor):
        print(f"Feature map {i+1} shape: {feat_map.shape}")
    else:
        print(f"Feature map {i+1} type: {type(feat_map)}")
