import copy
from torch import nn
import torch
import numpy as np
import math
from typing import Generator
import time

from packages.transformer.optimizer import *
from packages.transformer.data import *
from packages.transformer.encoder_decoder import *
from packages.transformer.utils import conv_bool_mask_to_neg_infty
   
def subsequent_mask(size):
    """Mask out subsequent positions"""
    attn_shape = (1, size, size) # NOTE: this is somewhat strange dimension; why the need for the 1?
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') 
    return torch.from_numpy(subsequent_mask) == 0

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time()  -start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def main():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0).cuda()
    model = make_model(V, V, N=2).cuda()
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for _ in range(10):
        model.train()
        run_epoch(cora_data_gen(V, 30, 20), model, 
                SimpleLossCompute(model.generator, criterion, model_opt))

        model.eval()
        print(run_epoch(cora_data_gen(V, 30, 5), model, 
                        SimpleLossCompute(model.generator, criterion, None)))

if __name__ == "__main__":
    main()