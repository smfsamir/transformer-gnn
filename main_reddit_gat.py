import time
import dgl
import numpy as np
from functools import partial
from torch.nn import CrossEntropyLoss
import torch
from typing import Iterator
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from packages.data_management.reddit_data import get_dataloader, load_reddit_adj, load_reddit_feats, load_reddit_labels, load_reddit_masks
from packages.transformer.optimizer import rate, SimpleLossCompute
from packages.utils.checkpointing import load_model, checkpoint_model
from packages.transformer.data import cora_data_gen, TransformerGraphBundleInput
from packages.transformer.encoder_decoder import make_model, EncoderDecoder

# NOTE: loading the dataset requires a significant amount of memory (30G+) for some reason; possibly because the dataset is loaded with an inefficient adjacency matrix. The COO format requires less than 1G. 
# dataset = dgl.data.RedditDataset()
# graph = dataset[0]

# adj_sparse = graph.adj(scipy_fmt='coo') 
# train_ids = np.arange(adj_sparse.shape[0])[graph.ndata['train_mask']]
# graph = dgl.graph((adj_sparse.row, adj_sparse.col))  
# dataset = None

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

# TODO: need to think about this carefully. We'll be evaluating on packed batches.
def eval_accuracy(subggraph_bundle_generator: Iterator[TransformerGraphBundleInput], model: EncoderDecoder):
    total = 0
    num_correct = 0 
    for graph_bundle in subggraph_bundle_generator:
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

def build_dataloader(graph: dgl.DGLHeteroGraph, sampler: dgl.dataloading.MultiLayerNeighborSampler, ids: torch.tensor):
    dataloader = dgl.dataloading.DataLoader(
        graph, ids, sampler,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=0)
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
    val_ids = all_ids[masks[0,:]]
    test_ids = all_ids[masks[0,:]]

    graph = dgl.graph((adj_sparse.row, adj_sparse.col))
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 5])
    _build_dataloader = partial(build_dataloader, graph, sampler)

    train_dataloader = _build_dataloader(train_ids)
    val_dataloader = _build_dataloader(val_ids)
    test_dataloader = _build_dataloader(test_ids)

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
    for nepoch in range(nepochs):
        model.train()
        nbatches = train_ids.shape[0] // batch_size
        epoch_loss, train_epoch_elapsed  = run_train_epoch(cora_data_gen(train_dataloader, nbatches, num_subgraphs, feats, labels, device), model, 
            SimpleLossCompute(model.generator, criterion), optimizer, lr_scheduler)

        tb_sw.add_scalar('Loss/train', epoch_loss, nepoch)
        tb_sw.add_scalar('Duration/train', train_epoch_elapsed, nepoch)
        
        model.eval()
        with torch.no_grad():
            validation_loss = run_eval_epoch(cora_data_gen(val_dataloader, nbatches, 1, feats, labels, device), model, 
                SimpleLossCompute(model.generator, criterion, None))
            tb_sw.add_scalar('Loss/validation', validation_loss, nepoch)
            if validation_loss < best_loss:
                checkpoint_model(model)
                best_loss = validation_loss
                best_loss_epoch = nepoch
train_model(32, 1)