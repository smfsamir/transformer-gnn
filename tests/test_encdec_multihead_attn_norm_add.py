import torch

import pytest 
from packages.transformer.apex_multihead_attn.encdec_multihead_attn import EncdecMultiheadAttn

def test_encdec_multihead_attn_norm_add() :
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    seq_length   = 80
    sequences    = 10
    hidden_dim   = 1024
    heads        = 16
    dropout_prob = 0.0

    ref_layer = EncdecMultiheadAttn(hidden_dim, 
                                            heads, 
                                            dropout=dropout_prob, 
                                            bias=False, 
                                            include_norm_add=True, 
                                            impl='default')
    ref_layer.cuda().half()
    ref_layer.reset_parameters()
    ref_inputs_q = torch.randn(seq_length, sequences, hidden_dim, 
                                    dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
    ref_inputs_k = torch.randn(seq_length, sequences, hidden_dim, 
                                    dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

    # Reset seed so parameters are identical
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    for _ in range(5) :
        ref_outputs,_ = ref_layer.forward(ref_inputs_q, 
                                                ref_inputs_k, 
                                                ref_inputs_k,
                                                key_padding_mask=None, 
                                                need_weights=False, 
                                                attn_mask=None,
                                                is_training=True)
        
        
        # ref_inputs_q.backward(grads)
    # self.assertTrue(torch.allclose(self.ref_inputs_q,  self.tst_inputs_q,  atol=1e-5, rtol=1e-5))
    # self.assertTrue(torch.allclose(self.ref_inputs_k,  self.tst_inputs_k,  atol=1e-5, rtol=1e-5))
    # self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
    # self.assertTrue(torch.allclose(self.ref_inputs_q.grad, self.tst_inputs_q.grad, atol=1e-3, rtol=1e-3))
