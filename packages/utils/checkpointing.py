import re
import torch
from torch.utils.checkpoint import checkpoint
import os
from collections import OrderedDict

def checkpoint_model(model, fname="model.pt", path="checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        {
            "model_state_dict": model.state_dict()
        },
        f"{path}/{fname}" 
    )

def load_model(model, fname="model.pt", path="checkpoints"):
    with open(f"{path}/{fname}", 'rb'):
        checkpoint = torch.load(
            f"{path}/{fname}"
        )
        return model.load_state_dict(checkpoint['model_state_dict']) 
    
def load_ddp_model(model, fname="model.pt", path="checkpoints_remote"):
    with open(f"{path}/{fname}", 'rb'):
        checkpoint = torch.load(
            f"{path}/{fname}"
        )
        checkpoint_state_dict = checkpoint['model_state_dict'] 
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in checkpoint_state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = checkpoint_state_dict
        model.load_state_dict(model_dict)
