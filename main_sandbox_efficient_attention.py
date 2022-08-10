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

# benchmark_jax_attention()
benchmark_standard_attention()
benchmark_subquad_attention_pt()
# benchmark_subquad_attention()
# benchmark_standard_attention()