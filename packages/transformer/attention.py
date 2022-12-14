import sys
import torch
from torch import nn
import torch.nn.functional as F
import math
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    # TODO: assuming the dimensions will work out here -- i.e., that minibatching works. Will confirm later.
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # assert len(query.shape) == 3
        # assert len(key.shape) == 3
        # assert len(value.shape) == 3

        # assert query.shape[0] == 1 and query.shape[-1] == self.d_model
        # assert len(mask.shape) == 3
        # assert mask.shape == (1, query.shape[1], query.shape[1])

        # if mask is not None: 
        #     mask = mask.unsqueeze(1) # NOTE: we shouldn't need to do this anymore, since our mask will become a 3D tensor :)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

def attention(query, key, value, mask=None, dropout=None):
    """TODO: what are the shapes of these?

    Args:
        query (_type_): _description_
        key (_type_): _description_
        value (_type_): _description_
        mask (_type_, optional): _description_. Defaults to None. TODO: what is the shape of the sparse matrix?
        dropout (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) 
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) 
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn