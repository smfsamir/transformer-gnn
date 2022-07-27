from typing import Iterator, Tuple, List
from dgl.heterograph import DGLBlock
import dgl
from dgl.data import citation_graph as citegrh
import sys
import numpy as np
import torch
# from torch_sparse import spmm

from packages.utils.sp_utils import convert_scipy_sparse_to_torch, select_submatrix

class TransformerGraphBundleInput:
    "Object for holding a minibatch of data with mask during training."
    def __init__(self, src_feats: torch.Tensor, trg_labels: torch.Tensor, src_mask: torch.Tensor, train_inds: torch.Tensor, ntokens: int, device: str):
        """Object for holding required (simple) graph data.

        Args:
            src_feats (torch.Tensor): Features for each node. Shape: B x B_in x D_feat. (To start, we'll use batch of size B=1).
            trg_labels (torch.Tensor): Target features for labelled nodes. Shape: B x B_out. (To start, we'll use minibatch of size B=1). 
            adj_mat (torch.Tensor): B x L x B_in x B_in. (To start, we'll use batch of size B=1). L is the number of layers, since each layer will have a different matrix.
            train_inds (torch.Tensor): B x B_out 
        """
        self.src_feats = src_feats.to(device)

        self.src_mask = src_mask

        self.trg_labels = trg_labels.to(device) 
        self.ntokens = ntokens # number of labelled tokens in the batch. TODO: this is WRONG when we do padding. we should pass it in explicitly.
        self.train_inds = train_inds

def retrieve_features_for_minibatch(batch_global_inds, all_features):
    """Retrieve features for the minibatch, ordered according to batch_specific_inds.

    Args:
        batch_global_inds (torch.Tensor): Indices in range 0...|V|-1
        batch_specific_inds (torch.Tensor): Indices in range 1...|B|-1 
        all_features (torch.Tensor): 2D tensor with |V| x D_input.
    """
    return all_features[batch_global_inds]

def retrieve_labels_for_minibatch(global_output_node_inds: torch.Tensor, all_labels: torch.Tensor):
    """Retrieve labels for the minibatch.

    Args:
        global_output_node_inds (torch.Tensor): _description_
        all_labels (torch.Tensor): _description_
    """
    return all_labels[global_output_node_inds] 

# TODO: shouldn't be necessary to convert back to dense. I was being lazy.
def convert_mfg_to_sg_adj(mfg: DGLBlock, square_shape: int, device: str):
    sparse_adj = mfg.adj()
    square_adj = torch.sparse_coo_tensor(sparse_adj._indices(), sparse_adj._values(), size=(square_shape, square_shape), device=device) 
    return square_adj.to_dense()
    # return square_adj

def construct_batch(target_nodes, subgraph_nodes, mfgs, all_features, all_labels, device):
    first_layer_mfg = mfgs[0]
    second_layer_mfg = mfgs[1]

    all_parallel_inds = torch.arange(subgraph_nodes.shape[0], device=device)
    first_layer_adj_submatrix = convert_mfg_to_sg_adj(first_layer_mfg, subgraph_nodes.shape[0], device) + torch.eye(subgraph_nodes.shape[0], device=device) 
    second_layer_adj_submatrix = convert_mfg_to_sg_adj(second_layer_mfg, subgraph_nodes.shape[0], device) + torch.eye(subgraph_nodes.shape[0], device=device) 
    output_node_inds = all_parallel_inds[: target_nodes.shape[0]]
    
    minibatch_adjacencies = torch.stack((first_layer_adj_submatrix, second_layer_adj_submatrix))
    all_minibatch_feats = all_features[subgraph_nodes, :]

    all_minibatch_feats = all_minibatch_feats.unsqueeze(0)
    minibatch_adjacencies = minibatch_adjacencies.unsqueeze(0) == 1
    minibatch_labels = all_labels[target_nodes].unsqueeze(0)
    output_node_inds = output_node_inds.unsqueeze(0)

    minibatch = TransformerGraphBundleInput(all_minibatch_feats, minibatch_labels, minibatch_adjacencies, output_node_inds, output_node_inds.shape[1], device)
    return minibatch

def pad_graph_bundle(graph_bundle: TransformerGraphBundleInput, device: str) -> None: # WARNING: mutates graph bundle object
    src_mask = graph_bundle.src_mask.squeeze(0) 
    size_subgraph = src_mask.shape[1]
    padded_src_mask = torch.zeros((src_mask.shape[0], 512, 512), device=device)
    padded_src_mask[:, : size_subgraph, : size_subgraph] = src_mask

    src_feats = graph_bundle.src_feats.squeeze(0)
    padded_src_feats = torch.zeros((512, src_feats.shape[-1]), device=device)
    padded_src_feats[: size_subgraph, :src_feats.shape[-1]] = src_feats
    graph_bundle.src_feats = padded_src_feats.unsqueeze(0)
    graph_bundle.src_mask = padded_src_mask.unsqueeze(0)

def stack_graph_bundles(graph_bundles: List[TransformerGraphBundleInput]) -> TransformerGraphBundleInput:
    src_masks = torch.cat([graph_bundle.src_mask for graph_bundle in graph_bundles])
    src_feats = torch.cat([graph_bundle.src_feats for graph_bundle in graph_bundles])
    trg_labels = torch.cat([graph_bundle.trg_labels for graph_bundle in graph_bundles])
    train_inds = torch.cat([graph_bundle.train_inds for graph_bundle in graph_bundles])
    ntokens = sum([graph_bundle.ntokens for graph_bundle in graph_bundles])
    return TransformerGraphBundleInput(src_feats, trg_labels, src_masks, train_inds, ntokens, 'cuda')

def cora_data_gen(dataloader: dgl.dataloading.DataLoader, 
                  nbatches: int,
                  num_subgraphs: int,
                  features: torch.Tensor, 
                  labels: torch.Tensor, 
                  device: str) -> TransformerGraphBundleInput:
    """Generate batches of cora datapoints one at a time, used for trainign and validation. Called once per epoch.

    Args:
        graph (dgl.DGLGraph): DGL graph
        train_nids (torch.Tensor): IDs in the graph that belong to the training set.
        neighbor_sampler (dgl.MultiLayerNeighborSampler): Neighbor sampler with fixed fanouts.
        batch_size (int): Number of datapoints in the batch.
        features (torch.Tensor): The features for all the data.
        labels (torch.Tensor): The labels for all the data.

    Returns:
        TransformerGraphBundleInput: 
    """
    dataloader_iter = iter(dataloader)
    niters = nbatches // num_subgraphs
    for _ in range(niters):
        if num_subgraphs > 1:
            graph_bundles = []
            for batch_i in range(num_subgraphs):
                input_nodes, output_nodes, mfgs = next(dataloader_iter) # input nodes gives us the requisite features. The mfgs gives us the requisite attention mask
                input_graph_bundle = construct_batch(output_nodes, input_nodes, mfgs, features, labels, device)
                pad_graph_bundle(input_graph_bundle, device) # mutation
                graph_bundles.append(input_graph_bundle)
            batch_input_graph_bundle = stack_graph_bundles(graph_bundles)
            yield batch_input_graph_bundle 
        else: 
            input_nodes, output_nodes, mfgs = next(dataloader_iter) # input nodes gives us the requisite features. The mfgs gives us the requisite attention mask
            input_graph_bundle = construct_batch(output_nodes, input_nodes, mfgs, features, labels, device)
            yield input_graph_bundle

def test_cora_data_gen(adj: torch.Tensor, features: torch.Tensor, test_nids: torch.Tensor, labels: torch.Tensor, device: str):
    adj_mat_layerwise = adj.expand(2,-1,-1) 
    return TransformerGraphBundleInput(features.unsqueeze(0), labels.unsqueeze(0), adj_mat_layerwise.unsqueeze(0), test_nids.unsqueeze(0), test_nids.shape[0], device)

def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    graph = dgl.from_networkx(data.graph) 
    adj = graph.adj(scipy_fmt='coo').toarray()
    graph = None
    return features, labels, train_mask, val_mask, test_mask, adj