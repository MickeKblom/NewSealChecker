import torch
from collections import OrderedDict

ckpt = torch.load("Models/CNN/best.pth", map_location="cpu")
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

clean = OrderedDict((k[7:] if k.startswith('module.') else k, v)
                    for k, v in state_dict.items())

torch.save(clean, "Models/CNN/best2.pth")
