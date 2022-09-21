import pdb
import math
from platform import node
from typing import Optional
import torch
from torch import nn
import numpy as np 
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

from .attention import MultiHeadedAttention
from .utils import batched_index_select

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, src_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
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
        # print(f"Processing subgraph of size: {batch_size}")
        # return node_embeds[torch.arange(batch_size), train_inds] # should have shape B x B_out x D.
        return batched_index_select(node_embeds, 1, train_inds)

class Generator(nn.Module):

    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return nn.LogSoftmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layers, fixed_output_dim):
        super(Encoder, self).__init__()
        self.num_layers = len(layers)
        self.layers = layers
        self.norm = nn.LayerNorm(fixed_output_dim + 2 * len(layers) * fixed_output_dim)
        
    def forward(self, x, masks):
        "Pass the input (and mask) through each layer in turn."
        layer_i = 0
        for layer in self.layers:
            x = layer(x, masks[:, layer_i, :, :])
            layer_i += 1
        return self.norm(x) # TODO: should we have this final layer norm...?
    
# class GraphLayerNorm(nn.Module):

#     def __init__(self, size)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size) # TODO: need to change this.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return torch.cat((x, self.dropout(sublayer(self.norm(x)))), -1)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, fixed_output_size, self_attn, feed_forward, dropout, layer_i):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = [(SublayerConnection(fixed_output_size + 2 * layer_i * fixed_output_size, dropout)), 
                         (SublayerConnection(fixed_output_size + (2 * layer_i + 1) * fixed_output_size, dropout))]
        # self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=mask)) 
        return self.sublayer[1](x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, e, dropout, layer_i):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model + (2 * layer_i + 1) * d_model, e * d_model) # TODO: change to be a function of layer_i
        self.w_2 = nn.Linear(e * d_model, d_model) # TODO: change to be a function of layer_i
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
               d_model: Optional[int]=512, e=4, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn_ins = partial(MultiHeadedAttention, h, d_model, dropout)
    ff_ins = partial(PositionwiseFeedForward, d_model, e, dropout)

    model = EncoderDecoder(
        Encoder([EncoderLayer(d_model, attn_ins(encoder_layer_i), ff_ins(encoder_layer_i), dropout, encoder_layer_i)  
                    for encoder_layer_i in range(N)], d_model),
        nn.Sequential(NodeEmbedding(d_model, d_input)),
        Generator(d_model + 2 * N * d_model, tgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal(p)
    return model

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)