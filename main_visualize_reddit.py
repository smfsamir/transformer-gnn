import pandas as pd
import torch
import dgl
import numpy as np
from packages.transformer.data import cora_data_gen

from packages.transformer.encoder_decoder import make_model
from packages.utils.checkpointing import load_ddp_model, load_model
from packages.data_management.reddit_data import load_reddit_adj, load_reddit_feats, load_reddit_labels, load_reddit_masks

import numpy as np
import scipy as sp
import torch
from torch.utils.data import IterableDataset
import json
import dgl

adj_sparse = load_reddit_adj()
feats = load_reddit_feats()
masks = load_reddit_masks()
labels = load_reddit_labels()
graph = dgl.graph((adj_sparse.row, adj_sparse.col))

model = make_model(feats.shape[1], len(torch.tensor(labels).unique()), N=2).cuda()
load_ddp_model(model, path="checkpoints")
all_ids = np.arange(masks.shape[1])
train_ids = all_ids[masks[0,:]]
fanouts = [5,5]

sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)

bs = 32
train_dataloader = dgl.dataloading.DataLoader(
    graph, train_ids, sampler,
    batch_size=bs,
    shuffle=True,
    drop_last=True,
    num_workers=0 
)

dataloader_iter = iter(train_dataloader)

target_num_embeddings = 1024
embeddings = []
max_graph_padding = bs + bs * fanouts[0] + (bs + bs * fanouts[0]) * fanouts[1] 

nbatches = target_num_embeddings // bs
device = 'cuda'
model.eval()
max_graph_padding = bs + bs * fanouts[0] + (bs + bs * fanouts[0]) * fanouts[1] 
with torch.no_grad():
    for graph_bundle in cora_data_gen(train_dataloader, nbatches, 1, feats, labels, max_graph_padding, device):
        out_embeds = model(graph_bundle.src_feats, graph_bundle.src_mask, graph_bundle.train_inds)
        embeddings.append(out_embeds)

all_embeds = torch.concat(embeddings).cpu().detach().numpy()
with open('reddit_vis_embeds.npy', 'wb') as f:
    np.save(f, all_embeds)