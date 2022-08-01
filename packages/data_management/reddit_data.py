import numpy as np
import scipy as sp
import torch
from torch.utils.data import IterableDataset
import json
import dgl

def inspect_graph(fname="data/reddit/reddit-G.json"):
    with open(fname, "r") as f:
        graph = json.load(f)
        pass

def get_dataloader(fanout: int, graph: dgl.graph, train_nids: torch.Tensor, batch_size: int, num_workers: int = 0):
    sampler = dgl.dataloading.MultiLayerNeighborSampler([fanout, fanout])
    dataloader = dgl.dataloading.NodeDataLoader(
        graph, train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0) # defaults to same device as the graph.
    return dataloader 

def load_reddit_adj(path="data/reddit_cpr", fname="reddit_adj_coo.npz"):
    adj = sp.sparse.load_npz(f'{path}/{fname}')
    return adj

def load_reddit_feats(path="data/reddit_cpr", fname="reddit_feats.npy"):
    with open(f"{path}/{fname}", "rb") as f:
        feats = np.load(f)
        return torch.tensor(feats)

def load_reddit_labels(path="data/reddit_cpr", fname="reddit_labels.npy"):
    with open(f"{path}/{fname}", "rb") as f:
        labels = np.load(f)
        return torch.tensor(labels) 

def load_reddit_masks(path="data/reddit_cpr", fname="reddit_masks.npy"):
    with open(f"{path}/{fname}", "rb") as f:
        masks = np.load(f)
        return torch.tensor(masks) 