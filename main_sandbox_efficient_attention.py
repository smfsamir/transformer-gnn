import chunk
import pdb
import math
import numpy as np
import torch
import time

from packages.efficient_attention.attention_torch import efficient_dot_product_attention
from packages.transformer.attention import attention
from memory_efficient_attention import efficient_dot_product_attention_jax

def benchmark_subquad_attention_pt():
    b = 1
    s = 289
    h = 8
    f = 64
    query = torch.rand((b, s, h, f), device='cuda', requires_grad=True)
    key = torch.rand((b, s, h, f), device='cuda')
    value = torch.rand((b, s, h ,f), device='cuda')

    mask = torch.rand((b, h, s, s), device='cuda') > 0.5

    chunk_size = round(np.sqrt(s))
    expanded_s = math.ceil(s / chunk_size) * chunk_size

    start = time.time()
    res = efficient_dot_product_attention(query, key, value, mask=mask, query_chunk_size=chunk_size,  key_chunk_size=chunk_size)
    elapsed = time.time() - start
    print(f"computing subquadratic attention took: {elapsed}")
    res.backward(torch.ones_like(query))
    # assert res.shape[1] == math.ceil(s / chunk_size) * chunk_size, f"{res.shape}"


    # final_chunk_inds = torch.arange(expanded_s - chunk_size, expanded_s)  
    # penultimate_chunk_inds = torch.arange(expanded_s - 2*chunk_size, expanded_s-chunk_size)
    # remainder = s % chunk_size

    # first_ind_final_chunk = final_chunk_inds[0]
    # crspnd_ind_penultimate_chunk = penultimate_chunk_inds[0+remainder] 
    # assert torch.equal(res[0,first_ind_final_chunk], res[0,crspnd_ind_penultimate_chunk]) # note: this is a key insight.

def benchmark_jax_attention():
    b = 1
    s = 302
    h = 8
    f = 64
    query = np.random.rand(b, s, h, f)
    key = np.random.rand(b, s, h, f)
    value = np.random.rand(b, s, h ,f)

    mask = np.random.rand(b, h, s, s)

    chunk_size = round(np.sqrt(s))
    expanded_s = math.ceil(s / chunk_size) * chunk_size

    start = time.time()
    res = efficient_dot_product_attention_jax(query, key, value, query_chunk_size=chunk_size,  key_chunk_size=chunk_size)
    elapsed = time.time() - start
    print(f"computing subquadratic attention took: {elapsed}")

def benchmark_standard_attention():
    b = 1
    s = 289
    h = 8
    f = 64
    query = torch.rand((b, h, s, f), device='cuda', requires_grad=True)
    key = torch.rand((b, h, s, f), device='cuda')
    value = torch.rand((b, h, s ,f), device='cuda')


    mask = torch.rand((b, s, s), device='cuda') > 0.5
    start = time.time()
    results = attention(query, key, value, mask)
    elapsed = time.time() - start
    print(f"Computing standard attention took: {elapsed}")
    results[0].backward(torch.ones_like(query))

def reimplement_attention():
    s = 289
    h = 8
    f = 64
    chunksize = int(np.sqrt(s))
    nc = s // chunksize
    num_queries = 1

    K_chunked = torch.rand(nc, chunksize, h, f)
    Q = torch.rand(1, h, f)
    
    single_key = torch.rand(h, f)
    single_q = torch.rand(h, f)
    single_key_res_bmm = torch.bmm(single_key.view(h, 1, f), single_q.view(h, f, 1))
    single_key_res_mm_sum = torch.mul(single_key, single_q).sum(axis=1)
    # assert torch.nn.functional.cosine_similarity(single_key_res_bmm.reshape(h), single_key_res_mm_sum, dim=0) == 1

    # so far, so good
    single_chunk = torch.rand(chunksize, h, f)
    single_q = torch.rand(1,h,f)
    single_chunk_mat_view = single_chunk.view(chunksize * h, f)

    single_q_expanded =single_q.expand(chunksize, -1, -1).reshape(chunksize*h, f)
    single_chunk_res = torch.bmm(single_chunk_mat_view.view(chunksize*h, 1, f), single_q_expanded.view(chunksize*h, f, 1))
    single_chunk_mm_res = torch.mul(single_chunk_mat_view, single_q_expanded).sum(axis=1) # works
    single_chunk_mm_res_2 = torch.mul(single_chunk, single_q).sum(axis=-1) # works


    # works.
    n_q = 2
    single_chunk = torch.rand(chunksize, h , f)
    single_chunk_repeated = single_chunk.repeat(1, 1, n_q)
    single_chunk_mat_view = single_chunk_repeated.view(chunksize * h, f * n_q)
    # single_q = torch.rand(n_q,h,f).reshape(1, h, n_q * f)
    chunk_q = torch.rand(n_q,h,f)
    chunk_q_expanded =chunk_q.view(1, h, n_q * f).expand(chunksize, -1, -1).reshape(chunksize*h, f * n_q)
    # single_chunk_res = torch.bmm(single_chunk_mat_view.view(chunksize*h, 1, f), single_q_expanded.view(chunksize*h, f, 1))

    pdb.set_trace()
    chunk_both_mm_res = torch.mul(single_chunk_mat_view, chunk_q_expanded).view(chunksize*h, n_q, f).sum(axis=-1)


def tensordot_batched():
    s = 289
    h = 8
    f = 64
    chunksize = int(np.sqrt(s))
    nc = s // chunksize

    single_key = torch.rand(h, f)
    single_q = torch.rand(h, f)
    single_key_res_td = torch.tensordot(single_key, single_q, dims=2)
    single_key_res_bmm = torch.bmm(single_key.view(h, 1, f), single_q.view(h, f, 1))
    pdb.set_trace()
    assert torch.equal(single_key_res_bmm, single_key_res_td)

    # single_chunk = torch.rand(chunksize, h, f)
    # single_q = torch.rand(h,f)

    # torch.tensordot(single_chunk, single_q, dims=-1)
reimplement_attention()


# tensordot_batched()


    # attn_weights = torch.matmul(single_chunk, )


