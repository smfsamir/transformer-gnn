import sys
from argparse import ArgumentParser
import time
import dgl
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp.autocast_mode import autocast
from dgl.data import citation_graph as citegrh
import torch
from main_transformer import TransformerGraphBundleInput
from typing import Iterator

from torch.utils.tensorboard import SummaryWriter
from packages.transformer.optimizer import *
from packages.transformer.encoder_decoder import *
from packages.transformer.utils import conv_bool_mask_to_neg_infty
from packages.transformer.data import TransformerGraphBundleInput, cora_data_gen, load_cora_data, test_cora_data_gen
from packages.utils.checkpointing import load_model, checkpoint_model
from packages.utils.inspect_attention import JacobianGAT, visualize_influence

SCALER = torch.cuda.amp.grad_scaler.GradScaler()

def run_epoch(
    subgraph_bundle_generator: Iterator[TransformerGraphBundleInput], 
    model: EncoderDecoder, 
    loss_compute: SimpleLossCompute,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR):
    "Standard Training and Logging Function"
    start = time.time()
    total_loss = 0
    ntokens = 0
    for subgraph_bundle in subgraph_bundle_generator: 
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out = model.forward(subgraph_bundle.src_feats, subgraph_bundle.src_mask,  
                                subgraph_bundle.train_inds) # B x B_out x model_D.  
            loss, loss_node = loss_compute(out, subgraph_bundle.trg_labels, subgraph_bundle.ntokens)
        ntokens += subgraph_bundle.ntokens 
        total_loss += loss
        SCALER.scale(loss_node).backward()
        # loss_node.backward()
        SCALER.step(optimizer)
        SCALER.update()
        scheduler.step()
    elapsed = time.time() - start
    print(f"Train loss on epoch: {total_loss / ntokens}; time taken: {elapsed}")
    return total_loss / ntokens

def run_eval_epoch(subgraph_bundle_generator: Iterator[TransformerGraphBundleInput], model: EncoderDecoder, \
                    loss_compute: SimpleLossCompute):
    start = time.time()
    total_loss = 0
    ntokens = 0
    for subgraph_bundle in subgraph_bundle_generator: 
        out = model.forward(subgraph_bundle.src_feats, subgraph_bundle.src_mask,  
                            subgraph_bundle.train_inds) # B x B_out x model_D.  
        # TODO: need to think about this loss computation carefully. Is it even possible?
        loss, _ = loss_compute(out, subgraph_bundle.trg_labels, subgraph_bundle.ntokens)
        ntokens += subgraph_bundle.ntokens 
        total_loss += loss
    elapsed = time.time() - start
    print(f"Validation loss on epoch: {total_loss / ntokens}")
    return total_loss / ntokens 

def eval_accuracy(graph_bundle: TransformerGraphBundleInput, model: EncoderDecoder):
    total = 0
    num_correct = 0 
    out = model.forward(graph_bundle.src_feats, graph_bundle.src_mask, graph_bundle.train_inds) # B x B_out x model_D.  
    out = model.generator(out) # B x num_nodes x num_classes
    out = out.squeeze(0) # num_nodes x num_classes
    out = out.argmax(axis=1) # num_nodes
    mb_test_labels = graph_bundle.trg_labels.squeeze(0)
    total += mb_test_labels.shape[0]
    num_correct += (out == mb_test_labels).sum()
    test_accuracy = num_correct / total
    print(test_accuracy)
    return test_accuracy

def train_model():
    tb_sw = SummaryWriter()

    data = citegrh.load_cora()
    features = torch.tensor(data.features, device='cuda')
    labels = torch.tensor(data.labels, device='cuda')
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    graph = data[0]
    adj = graph.adj(scipy_fmt='coo')
    graph = dgl.graph((adj.row, adj.col)).to('cuda')

    criterion = LabelSmoothing(size=8, padding_idx=7, smoothing=0.0).cuda()
    model = make_model(features.shape[1], len(labels.unique()) + 1, N=2).cuda() # +1 for the padding index, though I don't think it's necessary.
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)

    lr_scheduler = LambdaLR(
        optimizer = optimizer, lr_lambda=lambda step: rate(step, model.src_embed[0].d_model, factor = 1.0, warmup = 400)
    )

    nepochs = 100
    best_loss = float("inf") 
    train_nids = (torch.arange(0, graph.number_of_nodes())[train_mask]).to('cuda')
    val_nids = (torch.arange(0, graph.number_of_nodes())[val_mask]).to('cuda')
    test_nids = (torch.arange(0, graph.number_of_nodes())[test_mask]).to('cuda')
    for nepoch in range(nepochs):
        model.train()
        epoch_loss = run_epoch(cora_data_gen(graph, train_nids, 64, features, labels), model, 
            SimpleLossCompute(model.generator, criterion), optimizer, lr_scheduler)
        
        tb_sw.add_scalar('Loss/train', epoch_loss, nepoch)
        
        model.eval()
        with torch.no_grad():
            validation_loss = run_eval_epoch(cora_data_gen(graph, val_nids, 64, features, labels), model, 
                SimpleLossCompute(model.generator, criterion, None))
            tb_sw.add_scalar('Loss/validation', validation_loss, nepoch)
            if validation_loss < best_loss:
                checkpoint_model(model)
                best_loss = validation_loss
            
    load_model(model) # mutation
    model.eval()
    with torch.no_grad():
        test_labels = labels[test_nids]
        test_acc = eval_accuracy(test_cora_data_gen(graph.adj().to_dense().cuda() + torch.eye(adj.shape[0], device='cuda'), features, test_nids, test_labels), model)
        tb_sw.add_scalar('Accuracy/test', test_acc)

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
    elif args.test_dataloader_dgl:
        test_dataloader_dgl()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--visualize_jacobian", action="store_true")
    parser.add_argument("--test_dataloader_dgl", action="store_true")

    main(parser.parse_args())