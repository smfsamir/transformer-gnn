import pickle
import sys
from scipy.sparse import csr_matrix
import numpy as np

def build_large_csr():
    num_vertices = 10000
    num_edges = 21 * 10000 
    total_possible_edges = num_vertices ** 2

    arr = np.array([1] * num_edges + [0] * (total_possible_edges - num_edges))
    np.random.shuffle(arr)
    arr = arr.reshape(num_vertices, num_vertices)
    np.fill_diagonal(arr, 1)
    arr = csr_matrix(arr)
    print(arr.data.nbytes + arr.indptr.nbytes + arr.indices.nbytes )
    with open("data/benchmark_csr.pickle", "wb") as handle:
        pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_large_csr():
    with open("data/benchmark_csr.pickle", "rb") as handle:
        arr = pickle.load(handle)
    return arr

def collect_minibatch_indices(adj_mat, init_inds):
    # general

    # NOTE: we must do this when actually storing the values for a fast matrix. I suppose we'll have to do it during construction.
    # adj_mat.setdiag(1)

    two_hop_mat = adj_mat.multiply(adj_mat)
    one_and_two_hop_mat = adj_mat + two_hop_mat

    # specific to batch
    one_and_two_hop_mat_batch = one_and_two_hop_mat[init_inds,:] 
    one_and_two_hop_mat_batch_row_sum = one_and_two_hop_mat_batch.sum(axis=0)
    relevant_inds = one_and_two_hop_mat_batch_row_sum > 0
    return relevant_inds

def get_attn_mask(adj_mat, binary_mask):
    binary_mask.resize(10000,)
    adj_submatrix = adj_mat[binary_mask,:].tocsc()[:,binary_mask]
    return adj_submatrix 


# build_large_csr()
mat = load_large_csr()
bin_mask = collect_minibatch_indices(mat, [0, 1, 9998])
attn_mask = get_attn_mask(mat, bin_mask)


assert attn_mask.shape == (63,63)
# print(mat)