import math
from apex.contrib.multihead_attn import SelfMultiheadAttn
from typing import Optional
import torch
from torch import nn
import numpy as np 
import copy
import torch.nn.functional as F
from torch.autograd import Variable

from .attention import MultiHeadedAttention
from .utils import conv_bool_mask_to_neg_infty

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor, train_inds: Optional[torch.Tensor] = None):
        """
        Args:
            src (torch.Tensor): Features of each node in the graph. B x B_in x feat_D.
            src_mask (torch.Tensor): Adjacency matrices for the graph submatrices: B x L x B_in x B_in. NOTE: they have to be passed in padded already. So B_in really means max B_in
            train_inds (Optional[torch.Tensor], optional): Indices for the nodes that we're making predictions for: B x B_out.

        Returns:
            torch.Tensor: B x B_out x model_D. 
        """
        node_embeds = self.encoder(self.src_embed(src), src_mask) # should have shape B x B_in x D. But I really need to check this.
        batch_size = src.size(0)
        return node_embeds[torch.arange(batch_size), train_inds] # should have shape B x B_out x D.

class Generator(nn.Module):

    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return nn.LogSoftmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.num_layers = N
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, masks):
        "Pass the input (and mask) through each layer in turn."
        layer_i = 0
        for layer in self.layers:
            # x = layer(x, masks[:, layer_i, :, :])
            x = layer(x, masks.squeeze(0)[layer_i, :, :])
            layer_i += 1
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, attn_mask=mask)) 
        def _self_attn(input_x):
            input_x = input_x.transpose(1,0).contiguous()
            res = self.self_attn(x.transpose(1,0), x.transpose(1,0), x.transpose(1,0), attn_mask=mask)[0]
            res = res.transpose(1,0).contiguous()
            return res
        x = self.sublayer[0](x, _self_attn) 
        return self.sublayer[1](x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class NodeEmbedding(nn.Module):
    def __init__(self, d_model, d_input):
        super(NodeEmbedding, self).__init__()
        self.input_to_h = nn.Linear(d_input, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.input_to_h(x) * math.sqrt(self.d_model)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def make_model(d_input: int, tgt_vocab: int , N: Optional[int] = 6, 
               d_model: Optional[int]=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = SelfMultiheadAttn(d_model, h, dropout=0.1) # should bias be true? I think we essentially have it false.
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(NodeEmbedding(d_model, d_input)),
        nn.Sequential(Embeddings(d_model, tgt_vocab)),
        Generator(d_model, tgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)