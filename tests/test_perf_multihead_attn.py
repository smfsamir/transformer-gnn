import pytest
import torch

from packages.transformer.apex_multihead_attn.self_multihead_attn import SelfMultiheadAttn

def test_performance_self():
    seq_length = 64
    num_seqs_start = 10
    num_seqs_stop = 120
    num_seqs_inc = 5
    num_trials = 20
    warmup_trials = 5

    layers = 2
    hidden_dim = 1024
    heads = 16

    use_bias = False
    use_norm_add = False

    if not torch.cuda.is_available():
        raise NotImplementedError('Running on CPU is not supported')
    torch.cuda.set_device(0)

    torch.manual_seed(111)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(111)

    
    attn_layers = []
    for idx in range(0, layers) :
        attn_layers.append(SelfMultiheadAttn(hidden_dim, heads, dropout=0.1, bias=use_bias, include_norm_add=use_norm_add, impl='default'))
        attn_layers[idx].cuda()
        attn_layers[idx].half()
        attn_layers[idx].reset_parameters()
    start_evt_fwd = []
    start_evt_bwd = []
    stop_evt_bwd  = []
    for recorded_trial in range(0, num_trials) :
        start_evt_fwd.append(torch.cuda.Event(enable_timing=True))
        start_evt_bwd.append(torch.cuda.Event(enable_timing=True))
        stop_evt_bwd.append(torch.cuda.Event(enable_timing=True))
    
    for sequences in range(num_seqs_start, num_seqs_stop + num_seqs_inc, num_seqs_inc):
        inputs = torch.randn(seq_length, sequences, hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        grads = torch.randn_like(inputs)
   
        for trial in range(0, num_trials + warmup_trials) :
            layer_inputs  = inputs
            evt_idx       = trial - warmup_trials
        
            if evt_idx >= 0 :
                start_evt_fwd[evt_idx].record()
        
            for lyr_idx in range(0, layers) :
                outputs,_ = attn_layers[lyr_idx].forward(layer_inputs, 
                                                        layer_inputs, 
                                                        layer_inputs,
                                                        key_padding_mask=None, 
                                                        need_weights=False, 
                                                        attn_mask=None,
                                                        is_training=True)
                layer_inputs = outputs
        
            if evt_idx >= 0 :
                start_evt_bwd[evt_idx].record()

            # if not args.fwd :
            layer_inputs.backward(grads)
        
            if evt_idx >= 0 :
                stop_evt_bwd[evt_idx].record()
    torch.cuda.synchronize()

    elapsed_time_fwd = 0.0
    elapsed_time_bwd = 0.0
    for evt_idx in range(0, num_trials) :
        elapsed_time_fwd += start_evt_fwd[evt_idx].elapsed_time(start_evt_bwd[evt_idx])
        elapsed_time_bwd += start_evt_bwd[evt_idx].elapsed_time(stop_evt_bwd[evt_idx])

        print("[ {} Attn {} ]Total Tokens: {:4d} Sequences: {:3d} Sequence Length: {:3d} Fwd Time / Layer: {:.3f} ms Bwd Time / Layer: {:.3f} ms".format(
            'Self',              \
            'Norm&Add' if use_norm_add else '',                   \
            sequences*seq_length,                             \
            sequences,                                             \
            seq_length,                                       \
            elapsed_time_fwd / ( num_trials * layers ),      \
            elapsed_time_bwd / ( num_trials * layers )))