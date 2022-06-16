import sys
import numpy as np
import torch

class TransformerGraphBundleInput:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src: torch.Tensor, trg: torch.Tensor, adj_mat: torch.Tensor, train_mask: torch.Tensor):
        """Object for holding required (simple) graph data.

        Args:
            src (torch.Tensor): features for each node. Shape: 1 x num_nodes x num input feats. first dim is size 1 since batch size is 1 (for entire graph).
            trg (torch.Tensor): fe
            adj_mat (torch.Tensor): 
            train_mask (torch.Tensor): 
        """
        self.src = src.cuda()

        np.fill_diagonal(adj_mat, 1) # NOTE: be careful here. this mutates the adjacency matrix. we should pass in a copy

        self.src_mask = torch.from_numpy(adj_mat == 0).cuda()

        self.trg = trg.cuda() 
        self.train_mask = train_mask
        self.trg_mask = torch.from_numpy(np.full((trg.shape[1], 1), False, dtype=bool)).cuda() # TODO: we shouldn't be using this anywhere. 
        self.ntokens = 1 # since it's a classification problem

def cora_data_gen(cora_features: torch.Tensor, cora_labels: torch.Tensor, cora_train_mask: torch.Tensor,
                    cora_adj_mat: np.array) -> TransformerGraphBundleInput:
    """Generate batch with one entry at a time.

    Args:
        cora_features (torch.Tensor): 2D tensor containing (Publication x publication features)
        cora_labels (torch.Tensor): 1D tensor containing (Publication field)
        cora_train_mask (torch.Tensor): (1D tensor containing which publications should be used for the Training Set.)
        cora_adj_mat (torch.Tensor): 2D numpy array of the adjacency matrix.
    """
    cora_adj_mat = cora_adj_mat.copy()
    cora_train_labels = cora_labels[cora_train_mask]
    cora_train_labels = cora_train_labels.view(1, 1, cora_train_labels.shape[0]) # TODO: This could be wrong. Currently, I'm thinking it's B x seq_len x number of items.
    return TransformerGraphBundleInput(cora_features.unsqueeze(0), cora_train_labels, cora_adj_mat, cora_train_mask)