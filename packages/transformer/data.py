from cProfile import label
from lib2to3.pytree import convert
import dgl
from dgl.data import citation_graph as citegrh
import sys
import numpy as np
import torch
from torch_sparse import spmm

from packages.utils.sp_utils import convert_scipy_sparse_to_torch, select_submatrix

class TransformerGraphBundleInput:
    "Object for holding a minibatch of data with mask during training."
    def __init__(self, src_feats: torch.Tensor, trg_labels: torch.Tensor, adj_mat: torch.Tensor, train_inds: torch.Tensor):
        """Object for holding required (simple) graph data.

        Args:
            src_feats (torch.Tensor): Features for each node. Shape: B x B_in x D_feat. (To start, we'll use batch of size B=1).
            trg_labels (torch.Tensor): Target features for labelled nodes. Shape: B x B_out. (To start, we'll use minibatch of size B=1). 
            adj_mat (torch.Tensor): B x L x B_in x B_in. (To start, we'll use batch of size B=1). L is the number of layers, since each layer will have a different matrix.
            train_inds (torch.Tensor): B x B_out 
        """
        self.src_feats = src_feats.cuda()

        # TODO: should this be 0 or 1?
        self.src_mask = (adj_mat == 1) 

        self.trg_labels = trg_labels.cuda() 
        self.ntokens = trg_labels.shape[1] # number of labelled tokens in the batch. TODO: need to think about this. How will the minibatch updating work?
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

def cora_data_gen(graph: dgl.DGLGraph, train_nids: torch.Tensor, 
                  batch_size: int,
                  features: torch.Tensor, 
                  labels: torch.Tensor) -> TransformerGraphBundleInput:
    """Generate batches of cora datapoints one at a time, used for trainign and validation. Called once per epoch.

    Args:
        graph (dgl.DGLGraph): DGL graph
        train_nids (torch.Tensor): IDs in the graph that belong to the training set.
        neighbor_sampler (dgl.MultiLayerNeighborSampler): Neighbor sampler with fixed fanouts.
        batch_size (int): Number of datapoints in the batch.
        features (torch.Tensor): The features for all the data.
        labels (torch.Tensor): The labels for all the data.

    Returns:
        TransformerGraphBundleInput: _description_
    """
    # TODO: should these be constructed in the function? I presume not.
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 5])
    dataloader = dgl.dataloading.DataLoader(
        graph, train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0)
    dataloader_iter = iter(dataloader)
    sparse_adj = convert_scipy_sparse_to_torch(graph.adj(scipy_fmt='coo'))

    nbatches = len(train_nids) // batch_size
    for _ in range(nbatches):
        # TODO: must check that the node indices returned by the sampler align with those of the labels... 
            # should be fine, I think. If nec, come back to this when the model actually runs... 
        
        input_nodes, output_nodes, mfgs = next(dataloader_iter) # input nodes gives us the requisite features. The mfgs gives us the requisite attention mask
        all_parallel_indices = torch.arange(input_nodes.shape[0], device='cuda')

        src_inds_first_layer = (mfgs[0].srcdata[dgl.NID])
        dst_inds_first_layer = (mfgs[0].dstdata[dgl.NID])
        two_hop_neighbour_inds_argsort_inds = all_parallel_indices[dst_inds_first_layer.shape[0]:]
        output_node_argsort_inds = all_parallel_indices[: output_nodes.shape[0]] # NOTE: is this an invariant form for DGL? It's probably not guaranteed 

        first_layer_adj_submatrix = select_submatrix(sparse_adj, src_inds_first_layer, all_parallel_indices) # TODO: does this work?
        first_layer_adj_submatrix = first_layer_adj_submatrix + torch.eye(first_layer_adj_submatrix.shape[0], device='cuda') # NOTE: adding self-connections.

        second_layer_adj_submatrix = first_layer_adj_submatrix.detach().clone()
        second_layer_adj_submatrix[:, two_hop_neighbour_inds_argsort_inds] = 0 
        second_layer_adj_submatrix = second_layer_adj_submatrix + torch.eye(second_layer_adj_submatrix.shape[0], device='cuda') # NOTE: adding self-connections.
        
        minibatch_adjacencies = torch.stack((first_layer_adj_submatrix, second_layer_adj_submatrix))
        all_minibatch_feats = retrieve_features_for_minibatch(src_inds_first_layer, features)

        all_minibatch_feats = all_minibatch_feats.unsqueeze(0)
        minibatch_adjacencies = minibatch_adjacencies.unsqueeze(0)
        minibatch_labels = retrieve_labels_for_minibatch(output_nodes, labels).unsqueeze(0)
        output_node_inds = output_node_argsort_inds.unsqueeze(0)

        minibatch = TransformerGraphBundleInput(all_minibatch_feats, minibatch_labels, minibatch_adjacencies, output_node_inds)
        yield minibatch

def test_cora_data_gen(adj: torch.Tensor, features: torch.Tensor, test_nids: torch.Tensor, labels: torch.Tensor):
    # TODO: forgetting self connections
    adj_mat_layerwise = adj.expand(2,-1,-1) 
    return TransformerGraphBundleInput(features.unsqueeze(0), labels.unsqueeze(0), adj_mat_layerwise.unsqueeze(0), test_nids.unsqueeze(0))


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