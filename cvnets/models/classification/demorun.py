import torch
from cvnets.models.classification.mobilevit_v2 import MobileViTv2
from types import SimpleNamespace
import json

# Define the configuration using nested dictionaries.
# This structure matches the attributes accessed via 'getattr' in MobileViTv2 and its config.
opts_dict = {
    "model": {
        "classification": {
            # Accessed by MobileViTv2.__init__
            "n_classes": 1000,
            "mitv2": {
                # Accessed by get_configuration
                "width_multiplier": 1.0
            }
        },
        "layer": {
            # Accessed by MobileViTv2.__init__
            "global_pool": 'mean'
        }
    }
    # Add other keys if needed by base classes or specific layers
}

# Convert the nested dictionary to a SimpleNamespace object for attribute access (e.g., opts.model.layer.global_pool)
opts = json.loads(json.dumps(opts_dict), object_hook=lambda d: SimpleNamespace(**d))

# Instantiate the model using the created opts object
model = MobileViTv2(opts=opts)
model.eval() # Set the model to evaluation mode

# Create a dummy input image
img = torch.randn(1, 3, 256, 256)

# Pass the image through the model to get feature maps
feature_maps = model(img)

# Print the shape of each feature map
print(f"Received {len(feature_maps)} feature maps.")
for i, feat_map in enumerate(feature_maps):
    if isinstance(feat_map, torch.Tensor):
        print(f"Feature map {i+1} shape: {feat_map.shape}")
    else:
        print(f"Feature map {i+1} type: {type(feat_map)}")
