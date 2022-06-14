import time
import dgl
from dgl.data import citation_graph as citegrh
import torch
from main_transformer import TransformerGraphBundleInput
from packages.transformer.optimizer import *
from packages.transformer.encoder_decoder import *
from packages.transformer.utils import conv_bool_mask_to_neg_infty
from packages.transformer.data import TransformerGraphBundleInput, cora_data_gen



def run_epoch(graph_bundle: TransformerGraphBundleInput, model: EncoderDecoder, loss_compute: SimpleLossCompute):
    "Standard Training and Logging Function"
    start = time.time()

    out = model.forward(graph_bundle.src, graph_bundle.trg, 
                        graph_bundle.src_mask, graph_bundle.trg_mask)
    total_loss = loss_compute(out, graph_bundle.trg, graph_bundle.ntokens)
    elapsed = time.time()  -start
    print(f"Train loss on epoch: {total_loss}; time taken: {elapsed}")
    return total_loss 

def main():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    graph = dgl.from_networkx(data.graph) 
    adj = graph.adj(scipy_fmt='coo').to_array()
    graph = None

    criterion = LabelSmoothing(size=8, padding_idx=7, smoothing=0.0).cuda()
    model = make_model(features.shape[1], len(labels.unique()) + 1, N=1).cuda() # +1 for the padding index, though I don't think it's necessary.
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for _ in range(10):
        model.train()
        run_epoch(cora_data_gen(features, labels, mask, adj), model, 
        SimpleLossCompute(model.generator, criterion, model_opt))

if __name__ == "__main__":
    main()