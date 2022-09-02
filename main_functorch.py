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
num_neighbours = 10


single_q = torch.rand(1,h,f)
chunked_keys = all_keys.view(nc, chunksize, h, f)
dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

column_is = # array of size 289 x NUM_NEIGHB. Attain the column indices. Can we know what chunk we're in just by looking at the values?

def summarize_chunk(query, chunk, chunk_neighbour_inds, dummy): # chunk neighbour inds should tell us which items in the chunk NOT to mask. Possibly mask all of them...
    single_chunk_mul_res = torch.mul(chunk, query).sum(axis=-1) # works
    return single_chunk_mul_res

def scan_chunk(query, single_chunk, neighbour_inds):
    chunk_neighbour_inds = neighbour_inds  
    return checkpoint(summarize_chunk, query, single_chunk, neighbour_inds, dummy_tensor, use_reentrant=False)

def scan_queries(query, neighbour_inds_for_queries):
    res = vmap(partial(scan_chunk, query), in_dims=0)(chunked_keys, neighbour_inds_for_queries)
    return res

# TODO: change this, so each 
def generate_sparse_mask(): # get the indptr and the indices. 289 entries
    mask = rand(s, s, 0.1, format='csr', random_state=0)
    indptr = torch.tensor(mask.indptr)
    col_inds = torch.tensor(mask.indices)
    return indptr, col_inds

# sparse_mask_indptr, sparse_mask_inds = generate_sparse_mask()
# slice_indptr_left = sparse_mask_indptr[0:s]
# slice_indptr_right = sparse_mask_indptr[1:s+1]
column_indices = torch.randint(0, s, (s, num_neighbours))
column_indices = column_indices % chunksize # yuh
pdb.set_trace()

complete_res = vmap(scan_queries, in_dims=0)(all_queries, column_indices)
# rand_arr = torch.ones(3)

# def index_arr(ind):
#     print(ind)
#     return torch.index_select(rand_arr, 0, ind)

# tens_range = torch.arange(0,3).view(3,1)
# res = vmap(index_arr)(tens_range)