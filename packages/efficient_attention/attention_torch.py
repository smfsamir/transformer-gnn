import pdb
import torch
import torch.jit as jit
from torch.utils.checkpoint import checkpoint
from .utils import dynamic_slice, map_pt, scan
import math

@torch.jit.script
def _query_chunk_attention(query_idx, query, key, value,
                           mask, key_chunk_size=4096,
                           ):
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    num_q = query.shape[-3]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / math.sqrt(k_features)
    dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True )

    def summarize_chunk(key_idx, query, key, value, mask, bias, dummy):
        attn_weights = torch.einsum('...qhd,...khd->...qhk', query, key)
        big_neg = -1e9 
        big_neg = torch.tensor(big_neg, device=mask.device, dtype=torch.float32)
        mask = torch.einsum('...hqk->...qhk', mask)
        attn_weights = torch.where(mask, attn_weights, big_neg)
        max_score, _ = torch.max(attn_weights, -1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.einsum('...vhf,...qhv->...qhf', value, exp_weights)
        max_score = torch.einsum('...qhk->...qh', max_score)
        return exp_values, exp_weights.sum(dim=-1), max_score

    def chunk_scanner(chunk_idx):
        key_chunk = dynamic_slice(key, tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
                                  tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features))
        value_chunk = dynamic_slice(value, tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(value.shape[:-3]) + (key_chunk_size, num_heads, v_features))
        mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, 0, chunk_idx),
                                    tuple(mask.shape[:-3]) + (mask.shape[-3], mask.shape[-2], key_chunk_size))

        return checkpoint(summarize_chunk, chunk_idx, query, key_chunk, value_chunk, mask_chunk, bias_chunk, dummy_tensor)

    chunk_values, chunk_weights, chunk_max = map_pt(
        chunk_scanner, xs=torch.arange(0, num_kv, key_chunk_size))

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights

@torch.jit.script
def efficient_dot_product_attention(query, key, value,
                                    mask=None,
                                    query_chunk_size=1024,
                                    key_chunk_size=4096,
                                    ):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Note: query, key, value needn't have any batch dimensions.
      Args:
        query: queries for calculating attention with shape of
          `[batch..., q_length, num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of
          `[batch..., kv_length, num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of
          `[batch..., kv_length, num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          This can be used for incorporating padding masks, proximity bias, etc.
        mask: mask for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        query_chunk_size: int: query chunks size
        key_chunk_size: int: key chunks size
        bias_calc_fn: a bias calculation callback for each chunk, of form
          `(q_offset, k_offset, bias_chunk, attn_weights, calc_fn_data) -> bias`.
          This can be used for incorporating causal masks, padding masks,
          proximity bias, etc.
        mask_calc_fn: a mask calculation callback for each chunk, of form
          `(q_offset, k_offset, mask_chunk, attn_weights, calc_fn_data) -> mask`.
          This can be used for incorporating causal or other large masks.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        weights_calc_fn: a general attn_weights callback for each chunk, of form
          `(q_offset, k_offset, attn_weights, calc_fn_data) -> attn_weights`.
          attn_weights has shape of
          `[batch..., q_chunk_size, num_heads, k_chunk_size]`.
          This can be used to implement complex weights processing in a memory
          efficient way.
        calc_fn_data: optional pure data to pass to each per-chunk call of
          bias_calc_fn, mask_calc_fn, and weights_calc_fn.
        weights_calc_data: pure_data to pass with each call to weights_calc_fn
      Returns:
        Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
      """
    num_q, num_heads, q_features = query.shape[-3:]
    num_kv = key.shape[-3]

    actual_chunk_size = min(query_chunk_size, num_q)

    def chunk_scanner(chunk_idx, _):
        query_chunk = dynamic_slice(query, tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(query.shape[:-3]) + (actual_chunk_size, num_heads, q_features))

        mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, chunk_idx, 0),
                                    tuple(mask.shape[:-3]) + (mask.shape[-3], actual_chunk_size, mask.shape[-1]))

        return (chunk_idx + query_chunk_size,
                _query_chunk_attention(chunk_idx, query_chunk, key, value, mask_chunk, key_chunk_size=key_chunk_size,
                                       ))

    _, res = scan(chunk_scanner, init=0, xs=None, length=math.ceil(num_q / actual_chunk_size)) # TODO: this could be the line that has me fucked up. 
    rl = [res[i] for i in range(res.shape[0])]
    return torch.cat(rl, dim=-3)