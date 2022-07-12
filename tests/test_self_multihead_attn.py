import torch

import pytest 

from packages.transformer.apex_multihead_attn.self_multihead_attn import SelfMultiheadAttn

HIDDEN_DIM = 1024

# @pytest.fixture
# def ref_layer():
#     yield ref_layer

# @pytest.fixture
# def ref_inputs():
    
    # yield torch.randn(seq_length, sequences, HIDDEN_DIM, 
    #                                 dtype=torch.float).requires_grad_(True)
def test_self_multihead_attn():
    seq_length   = 10
    sequences    = 2 
    ref_inputs = torch.randn(seq_length, sequences, HIDDEN_DIM, 
                                        dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
    heads        = 16
    dropout_prob = 0.0
    ref_layer = SelfMultiheadAttn(HIDDEN_DIM, 
                                        heads, 
                                        dropout=dropout_prob, 
                                        bias=False, 
                                        include_norm_add=False, 
                                        impl='default')
    ref_layer.cuda().half()
    # ref_layer
    ref_layer.reset_parameters()
    grads         = torch.randn_like(ref_inputs)

    ref_outputs,_ = ref_layer.forward(ref_inputs, 
                                            ref_inputs, 
                                            ref_inputs,
                                            key_padding_mask=None, 
                                            need_weights=False, 
                                            attn_mask=None,
                                            is_training=True)

    # ref_inputs.backward(grads)
    # self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,  atol=1e-5, rtol=1e-5))
    # self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
    # self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad, atol=1e-3, rtol=1e-3))

# def test_self_multihead_attn_time_mask(ref_layer, ref_inputs):
#     grads         = torch.randn_like(ref_inputs)
#     time_mask_byte= torch.triu(torch.ones(ref_inputs.size(0), ref_inputs.size(0), device=torch.device("cuda"), dtype=torch.uint8), 1)
#     # time_mask_byte= torch.triu(torch.ones(ref_inputs.size(0), ref_inputs.size(0), dtype=torch.uint8), 1)
#     time_mask_bool= time_mask_byte.to(torch.bool)

#     ref_outputs,_ = ref_layer.forward(ref_inputs, 
#                                             ref_inputs, 
#                                             ref_inputs,
#                                             key_padding_mask=None, 
#                                             need_weights=False, 
#                                             attn_mask=time_mask_bool,
#                                             is_training=True)
    
#     ref_inputs.backward(grads)

    # self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,  atol=1e-5, rtol=1e-5))
    # self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
    # self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad, atol=1e-3, rtol=1e-3))

# def test_self_multihead_attn_pad_mask(ref_inputs, ref_layer) :
#     grads         = torch.randn_like(ref_inputs)
#     pad_mask_byte = torch.tril(torch.ones(ref_inputs.size(1), ref_inputs.size(0), dtype=torch.uint8), 1)
#     pad_mask_bool = pad_mask_byte.to(torch.bool)

#     ref_outputs,_ = ref_layer.forward(ref_inputs, 
#                                             ref_inputs, 
#                                             ref_inputs,
#                                             key_padding_mask=pad_mask_bool, 
#                                             need_weights=False, 
#                                             attn_mask=None,
#                                             is_training=True)

    
#     ref_inputs.backward(grads)
#     # self.tst_inputs.backward(grads)

#     # self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,  atol=1e-5, rtol=1e-5))
#     # self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
#     # self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad, atol=1e-3, rtol=1e-3))