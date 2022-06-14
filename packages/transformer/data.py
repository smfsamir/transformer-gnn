import numpy as np
import torch

class TransformerGraphBundleInput:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg, adj_mat):
        self.src = src.cuda()

        np.fill_diagonal(adj_mat, 1) # NOTE: be careful here. this mutates the adjacency matrix. we should pass in a copy

        self.src_mask = torch.from_numpy(adj_mat == 0).cuda()

        self.trg = trg.cuda()
        self.trg_mask = torch.from_numpy(np.full((trg.shape[0], 1), False, dtype=bool)).cuda()
        self.ntokens = 1 # since it's a classification problem

def cora_data_gen(cora_features: torch.Tensor, cora_labels: torch.Tensor, cora_train_mask: torch.Tensor,
                    cora_adj_mat: np.array) -> TransformerGraphBundleInput:
    """Generate synthetic data from multilingual BERT.

    Args:
        cora_features (torch.Tensor): 2D tensor containing (Publication x publication features)
        cora_labels (torch.Tensor): 1D tensor containing (Publication field)
        cora_train_mask (torch.Tensor): (1D tensor containing which publications should be used for the Training Set.)
        cora_adj_mat (torch.Tensor): 2D numpy array of the adjacency matrix.
    """
    cora_adj_mat = cora_adj_mat.copy()
    cora_labels = cora_labels.unsqueeze(-1)
    return TransformerGraphBundleInput(cora_features, cora_labels, cora_adj_mat)