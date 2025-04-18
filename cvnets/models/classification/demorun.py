import torch
from cvnets.models.classification.mobilevit_v2 import MobileViTv2
from cvnets.models.classification.config import mobilevit_v2
from cvnets import utils

opts = utils.AttrDict()
opts.model = utils.AttrDict()
opts.model.classification = utils.AttrDict()
opts.model.classification.n_classes = 1000 # doesn't matter since we are extracting feature maps
opts.model.layer = utils.AttrDict()
opts.model.layer.global_pool = 'mean'

# you can set the width multiplier in the config file
# or you can pass it as an argument
# opts.model.classification.mobilevitv2.width_multiplier = 1.0

model = MobileViTv2(opts=opts)
model.eval() # set the model to evaluation mode

# create a dummy input image
img = torch.randn(1, 3, 256, 256)

# pass the image through the model
feature_maps = model(img)

# print the shape of each feature map
for i, feat_map in enumerate(feature_maps):
    print(f"Feature map {i+1} shape: {feat_map.shape}")