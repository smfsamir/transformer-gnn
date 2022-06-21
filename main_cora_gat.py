from argparse import ArgumentParser
import time
import dgl
from dgl.data import citation_graph as citegrh
import torch
from main_transformer import TransformerGraphBundleInput
from torch.utils.tensorboard import SummaryWriter
from packages.transformer.optimizer import *
from packages.transformer.encoder_decoder import *
from packages.transformer.utils import conv_bool_mask_to_neg_infty
from packages.transformer.data import TransformerGraphBundleInput, cora_data_gen, load_cora_data
from packages.utils.checkpointing import load_model, checkpoint_model
from packages.utils.inspect_attention import JacobianGAT, visualize_influence

def run_epoch(graph_bundle: TransformerGraphBundleInput, model: EncoderDecoder, loss_compute: SimpleLossCompute):
    "Standard Training and Logging Function"
    start = time.time()

    out = model.forward(graph_bundle.src, graph_bundle.trg, 
                        graph_bundle.src_mask, graph_bundle.trg_mask, 
                        graph_bundle.train_mask) # shape: B x D x N, where N is number of nodes we're making predictions for.
    # graph_bundle.trg is (B x 1 x N). Not sure if correct. The 1 stands for the seq_len.
    total_loss = loss_compute(out, graph_bundle.trg, graph_bundle.ntokens)
    elapsed = time.time() - start
    ntokens = graph_bundle.ntokens 
    print(f"Train loss on epoch: {total_loss / ntokens}; time taken: {elapsed}")
    return total_loss / ntokens

def run_eval_epoch(graph_bundle: TransformerGraphBundleInput, model: EncoderDecoder, \
                    loss_compute: SimpleLossCompute):
    out = model.forward(graph_bundle.src, graph_bundle.trg, 
                    graph_bundle.src_mask, graph_bundle.trg_mask, 
                    graph_bundle.train_mask) # shape: B x D x N, where N is number of nodes we're making predictions for.
    total_loss = loss_compute(out, graph_bundle.trg, graph_bundle.ntokens)
    ntokens = graph_bundle.ntokens 
    print(f"Validation loss on epoch: {total_loss / ntokens}")
    return total_loss / ntokens 

def eval_accuracy(graph_bundle: TransformerGraphBundleInput, model: EncoderDecoder):
    out = model.forward(graph_bundle.src, graph_bundle.trg, 
                    graph_bundle.src_mask, graph_bundle.trg_mask, 
                    graph_bundle.train_mask) # shape: B x D x N, where N is number of nodes we're making predictions for.
    out = model.generator(out)
    out = out.squeeze(0) 
    out = out.argmax(axis=1)
    test_labels = graph_bundle.trg.view(1000)
    # print(f"Test accuracy out shape: {out.shape}")
    # print(f"Test accuracy labels shape: {test_labels}")
    total = len(test_labels)
    correct = (out == test_labels).sum()
    test_accuracy = correct/total
    print(f"Test accuracy: {correct/total}")
    return test_accuracy

def train_model():
    tb_sw = SummaryWriter()

    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    graph = dgl.from_networkx(data.graph) 
    adj = graph.adj(scipy_fmt='coo').toarray()
    
    graph = None

    criterion = LabelSmoothing(size=8, padding_idx=7, smoothing=0.0).cuda()
    model = make_model(features.shape[1], len(labels.unique()) + 1, N=2).cuda() # +1 for the padding index, though I don't think it's necessary.
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    nepochs = 200
    best_loss = float("inf") 
    for nepoch in range(nepochs):
        model.train()
        epoch_loss = run_epoch(cora_data_gen(features, labels, train_mask, adj), model, 
            SimpleLossCompute(model.generator, criterion, model_opt))
        tb_sw.add_scalar('Loss/train', epoch_loss, nepoch)
        
        model.eval()
        with torch.no_grad():
            validation_loss = run_eval_epoch(cora_data_gen(features, labels, val_mask, adj), model, 
                SimpleLossCompute(model.generator, criterion, None))
            tb_sw.add_scalar('Loss/validation', validation_loss, nepoch)

            if validation_loss < best_loss:
                checkpoint_model(model)
                best_loss = validation_loss
            
    load_model(model) # mutation
    model.eval()
    with torch.no_grad():
        test_acc = eval_accuracy(cora_data_gen(features, labels, test_mask, adj), model)
        # tb_sw.add_scalar('Accuracy/test', test_acc)

def visualize_jacobian():
    features, labels, train_mask, _, _, adj = load_cora_data()
    model = make_model(features.shape[1], len(labels.unique()) + 1, N=2).cuda() # +1 for the padding index, though I don't think it's necessary.
    load_model(model) # mutation
    jacobianized_model = JacobianGAT(model, 0, 1, cora_data_gen(features, labels, train_mask, adj))
    visualize_influence(jacobianized_model, features[0])

def main(args):
    if args.train_model:
        train_model()
    elif args.visualize_jacobian:
        visualize_jacobian()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--visualize_jacobian", action="store_true")

    main(parser.parse_args())