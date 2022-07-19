import torch
import numpy as np

def conv_bool_mask_to_neg_infty(bool_mask: torch.bool) -> torch.float:
    new_attn_mask = torch.zeros_like(bool_mask, dtype=float)
    new_attn_mask.masked_fill_(bool_mask, float("-inf"))
    return new_attn_mask

def batched_index_select(input, dim, index):
	views = [input.shape[0]] + \
		[1 if i != dim else -1 for i in range(1, len(input.shape))]
	expanse = list(input.shape)
	expanse[0] = -1
	expanse[dim] = -1
	index = index.view(views).expand(expanse)
	return torch.gather(input, dim, index)