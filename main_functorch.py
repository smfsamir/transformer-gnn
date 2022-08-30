from scipy.sparse import rand
import numpy as np
import time
import torch
import pdb
from functorch import vmap
from functools import partial
from torch.utils.checkpoint import checkpoint

s = 289
h = 8
f = 64
chunksize = int(np.sqrt(s))
nc = s // chunksize
num_queries = 1

all_keys = torch.rand(s, h, f)
all_queries = torch.rand(s,h,f)


single_q = torch.rand(1,h,f)
chunked_keys = all_keys.view(nc, chunksize, h, f)
dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)


def summarize_chunk(query, chunk, dummy):
    single_chunk_mul_res = torch.mul(chunk, query).sum(axis=-1) # works
    mask = torch.zeros_like(single_chunk_mul_res)
    return single_chunk_mul_res

def scan_chunk(query, single_chunk):
    return checkpoint(summarize_chunk, query, single_chunk, dummy_tensor, use_reentrant=False)

def scan_queries(all_col_inds, slice_left_ind, slice_right_ind, query):
    # slice_range = torch.arange(slice_left_ind, slice_right_ind)
    # nonzero_cols = torch.index_select(all_col_inds,0,slice_range) # assuming this works, we have to pass it into the partial.
    nonzero_cols = torch.narrow(all_col_inds, slice_left_ind, slice_right_ind)
    print(nonzero_cols)
    res = vmap(partial(scan_chunk, query), in_dims=0)(chunked_keys)
    return res

# TODO: change this, so each 
def generate_sparse_mask(): # get the indptr and the indices. 289 entries
    mask = rand(s, s, 0.1, format='csr', random_state=0)
    indptr = torch.tensor(mask.indptr)
    col_inds = torch.tensor(mask.indices)
    return indptr, col_inds

sparse_mask_indptr, sparse_mask_inds = generate_sparse_mask()
slice_indptr_left = sparse_mask_indptr[0:s]
slice_indptr_right = sparse_mask_indptr[1:s+1]
pdb.set_trace()

complete_res = vmap(partial(scan_queries, sparse_mask_inds), in_dims=0)(all_queries, slice_indptr_left, slice_indptr_right )


# rand_arr = torch.ones(3)
# def index_arr(ind):
#     print(ind)
#     return torch.index_select(rand_arr, 0, ind)

# tens_range = torch.arange(0,3).view(3,1)
# res = vmap(index_arr)(tens_range)



