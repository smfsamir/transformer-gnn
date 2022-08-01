import argparse
import time
import dgl
import numpy as np
from functools import partial
from torch.nn import CrossEntropyLoss
import torch
from typing import Iterator
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from typing import List
from sklearn.metrics import f1_score

from packages.data_management.reddit_data import get_dataloader, load_reddit_adj, load_reddit_feats, load_reddit_labels, load_reddit_masks
from packages.transformer.optimizer import rate, SimpleLossCompute
from packages.utils.checkpointing import load_model, checkpoint_model
from packages.transformer.data import cora_data_gen, TransformerGraphBundleInput
from packages.transformer.encoder_decoder import make_model, EncoderDecoder


def run_train_epoch(subgraph_bundle_generator: Iterator[TransformerGraphBundleInput], model: EncoderDecoder, loss_compute: SimpleLossCompute,
                    optimizer: torch.optim.Optimizer, scheduler: LambdaLR):
    "Standard Training and Logging Function"
    start = time.time()
    total_loss = 0
    ntokens = 0
    for subgraph_bundle in subgraph_bundle_generator: 
        optimizer.zero_grad(set_to_none=True)
        out = model.forward(subgraph_bundle.src_feats, subgraph_bundle.src_mask,  
                            subgraph_bundle.train_inds) # B x B_out x model_D.  
        # TODO: need to think about this loss computation carefully. Is it even possible?
        loss, loss_node = loss_compute(out, subgraph_bundle.trg_labels, subgraph_bundle.ntokens)

        ntokens += subgraph_bundle.ntokens 
        total_loss += loss
        loss_node.backward()
        optimizer.step()
        scheduler.step()
    elapsed = time.time() - start
    print(f"Train loss on epoch: {total_loss / ntokens}; time taken: {elapsed}")
    return total_loss / ntokens, elapsed

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

# TODO: need to think about this carefully if we use B > 1. We'll be evaluating on packed batches later.
def eval_accuracy(subggraph_bundle_generator: Iterator[TransformerGraphBundleInput], model: EncoderDecoder):
    total = 0
    num_correct = 0 
    predictions = []
    test_labels = []
    for graph_bundle in subggraph_bundle_generator:
        out = model.forward(graph_bundle.src_feats, graph_bundle.src_mask, graph_bundle.train_inds) # B x B_out x model_D.  
        out = model.generator(out) # B x num_nodes x num_classes
        out = out.squeeze(0) # num_nodes x num_classes
        out = out.argmax(axis=1) # num_nodes
        mb_test_labels = graph_bundle.trg_labels.squeeze(0)
        # total += mb_test_labels.shape[0]
        # num_correct += (out == mb_test_labels).sum()
        predictions.append(out)
        test_labels.append(mb_test_labels)
    predictions = torch.cat(predictions)
    test_labels = torch.cat(test_labels)
    # test_accuracy = num_correct / total
    test_f1 = f1_score(test_labels.cpu(), predictions.cpu(), average='micro')
    return test_f1

def build_dataloader(graph: dgl.DGLHeteroGraph, bs: int, fanouts: List[int], ids: torch.tensor):
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.DataLoader(
        graph, ids, sampler,
        batch_size=bs,
        shuffle=True,
        drop_last=True
    )
    return dataloader

def train_model(bs: int, num_sg: int):
    """Train the model on Reddit data.

    Args:
        bs (int): _description_
        num_sg (int): _description_
    """
    adj_sparse = load_reddit_adj()
    feats = load_reddit_feats()
    masks = load_reddit_masks()
    labels = load_reddit_labels()
    num_classes = len(labels.unique())
    all_ids = np.arange(masks.shape[1])
    train_ids = all_ids[masks[0,:]]
    val_ids = all_ids[masks[1,:]]
    test_ids = all_ids[masks[2,:]]

    graph = dgl.graph((adj_sparse.row, adj_sparse.col))
    fanouts = [5,5]
    _build_dataloader = partial(build_dataloader, graph, bs, fanouts)

    train_dataloader = _build_dataloader(train_ids)
    val_dataloader = _build_dataloader(val_ids)
    # test_dataloader = _build_dataloader(test_ids)

    criterion = CrossEntropyLoss(reduction='sum').cuda()
    model = make_model(feats.shape[1], num_classes, N=2).cuda() 

    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-6)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, 512, factor=1, warmup=3000
        ),
    )
    batch_size = bs
    num_subgraphs = num_sg

    tb_log_dir = f"runs/batch-{batch_size}_num_sg-{num_subgraphs}_reddit"
    tb_sw = SummaryWriter(tb_log_dir)

    device= 'cuda'
    nepochs = 30
    best_loss = float('inf')
    max_graph_padding = batch_size + batch_size * fanouts[0] + (batch_size + batch_size * fanouts[0]) * fanouts[1] 
    for nepoch in range(nepochs):
        model.train()
        train_nbatches = train_ids.shape[0] // batch_size
        epoch_loss, train_epoch_elapsed  = run_train_epoch(cora_data_gen(train_dataloader, train_nbatches, num_subgraphs, feats, labels, max_graph_padding, device), model, SimpleLossCompute(model.generator, criterion), optimizer, lr_scheduler)

        tb_sw.add_scalar('Loss/train', epoch_loss, nepoch)
        tb_sw.add_scalar('Duration/train', train_epoch_elapsed, nepoch)
        
        model.eval()
        val_nbatches = val_ids.shape[0] // batch_size 
        with torch.no_grad():
            validation_loss = run_eval_epoch(cora_data_gen(val_dataloader, val_nbatches, num_subgraphs, feats, labels, max_graph_padding, device), model, 
                SimpleLossCompute(model.generator, criterion))
            tb_sw.add_scalar('Loss/validation', validation_loss, nepoch)
            if validation_loss < best_loss:
                checkpoint_model(model)
                best_loss = validation_loss
                best_loss_epoch = nepoch

def evaluate_model(bs, num_subgraphs):
    adj_sparse = load_reddit_adj()
    feats = load_reddit_feats()
    masks = load_reddit_masks()
    labels = load_reddit_labels()
    num_classes = len(labels.unique())
    all_ids = np.arange(masks.shape[1])
    test_ids = all_ids[masks[2,:]]

    graph = dgl.graph((adj_sparse.row, adj_sparse.col))
    _build_dataloader = partial(build_dataloader, graph, bs, [5,5])
    test_dataloader = _build_dataloader(test_ids)
    model = make_model(feats.shape[1], num_classes, N=2).cuda() 
    load_model(model) # mutation
    model.eval()
    device = 'cuda'
    with torch.no_grad():
        test_nbatches = test_ids.shape[0] // bs 
        test_acc = eval_accuracy(cora_data_gen(test_dataloader, test_nbatches, num_subgraphs, feats, labels, device), model)
    print(f"{test_acc:.3f}")



def main(args):
    if args.evaluate_model:
        evaluate_model(args.bs, args.num_sg)
    elif args.train_model:
        train_model(args.bs, args.num_sg)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bs", type=int)
    parser.add_argument("num_sg", type=int)
    parser.add_argument("--train_model", action='store_true')
    parser.add_argument("--evaluate_model", action='store_true')

    main(parser.parse_args())