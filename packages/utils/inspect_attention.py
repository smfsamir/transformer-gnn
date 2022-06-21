import torch
from torch.autograd.functional import jacobian
from packages.transformer.data import TransformerGraphBundleInput

from packages.transformer.encoder_decoder import EncoderDecoder
# def visualize_attn_entropy(model, graph):
#     # TODO: visualize attention for each distribution for each head. easy
#     pass

class JacobianGAT:
    def __init__(self, model: EncoderDecoder, node_i_ind: int, node_j_ind: int, graph_bundle: TransformerGraphBundleInput):
        self.model = model
        self.graph_bundle = graph_bundle # |V| x D_IN
        self.node_i_ind = node_i_ind
        self.node_j_ind = node_j_ind

    def __call__(self, node_i_x: torch.Tensor):
        self.graph_bundle.src[0, self.node_i_ind, :] = node_i_x # may have to account for the batching here
        out = self.model(self.graph_bundle.src, self.graph_bundle.trg, 
                    self.graph_bundle.src_mask, self.graph_bundle.trg_mask, 
                    self.graph_bundle.train_mask) # |V| x D_H, I think. Possibly need to unsqueeze or something. Test it out.
        return out[0, self.node_j_ind, :]  # D_H, I think. may have to account for batching 

def visualize_influence(model: JacobianGAT, node_i_input: torch.Tensor):
    func_node_j_output = lambda node_i_x: model(node_i_x)
    jac_eval = jacobian(func_node_j_output, node_i_input)
    print(jac_eval)

