
import os, argparse
from lib.GLCONet_Swin import Network
import torch
from thop import profile


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Model
print('==> Building model..')
input_features = torch.randn(1, 3, 384, 384)
model = Network(128)
#dummy_input = torch.randn(1, 2048, 12, 12)
flops, params = profile(model, (input_features,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))


