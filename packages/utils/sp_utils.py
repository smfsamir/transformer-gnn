import numpy as np
from numpy import diagonal
import torch 
from scipy.sparse import coo_matrix
"""Utilities for manipulating COO sparse matrices
(since these are the only ones supported by CUDA). :)
"""
def select_submatrix(sparse_adj: torch.Tensor, indices: torch.Tensor, indices_argsort: torch.Tensor):
    """_summary_

    Args:
        sparse_adj (torch.Tensor)
        indices (torch.Tensor): the rows and columns we want from the adjacency matrix
        indices_argsort (torch.Tensor): the indices that would sort the array.  
    
    Returns:
        torch.Tensor: Dense square submatrix containing the elements we want from the adjacency matrix.
    """
    # NOTE: if necessary, this could be faster by changing the row selection matrix and column matrix to sparse matrices.
    assert sparse_adj.is_sparse
    row_diagonal_inds = torch.vstack((indices_argsort, indices))
    vals = torch.ones(indices.shape[0], device='cuda')
    row_selection_mat = torch.sparse_coo_tensor(row_diagonal_inds, vals, size=(indices.shape[0], sparse_adj.shape[1]), device='cuda') 

    col_diagonal_inds = torch.vstack((indices, indices_argsort))
    col_selection_mat = torch.sparse_coo_tensor(col_diagonal_inds, vals, size=(sparse_adj.shape[0], indices.shape[0]), device='cuda') # dense
    row_selected_mat = torch.sparse.mm(row_selection_mat, sparse_adj)  
    submat = torch.sparse.mm(row_selected_mat, col_selection_mat)
    return submat.to_dense() # NOTE: It's possible we could just keep this as sparse. I doubt it'll make a difference

def convert_scipy_sparse_to_torch(scipy_coo_mat):
    indices = np.vstack((scipy_coo_mat.row, scipy_coo_mat.col))
    values = scipy_coo_mat.data
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scipy_coo_mat.shape 
    return torch.sparse_coo_tensor(i, v, (shape), device='cuda')    