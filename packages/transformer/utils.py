import torch
import numpy as np

def conv_bool_mask_to_neg_infty(bool_mask: torch.bool) -> torch.float:
    new_attn_mask = torch.zeros_like(bool_mask, dtype=float)
    new_attn_mask.masked_fill_(bool_mask, float("-inf"))
    return new_attn_mask
