import gc
import pdb
from hyperopt import hp, fmin, tpe, space_eval
from argparse import ArgumentParser
from functools import partial
from typing import Optional, List
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import dgl
from dgl.data import citation_graph as citegrh
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from main_transformer import TransformerGraphBundleInput
from typing import Iterator
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from packages.transformer.optimizer import *
from packages.transformer.encoder_decoder import *
from packages.transformer.utils import conv_bool_mask_to_neg_infty
from packages.transformer.data import TransformerGraphBundleInput, cora_data_gen, load_cora_data, test_cora_data_gen
from packages.utils.checkpointing import load_model, checkpoint_model
from packages.utils.inspect_attention import JacobianGAT, visualize_influence

def run_train_epoch(subgraph_bundle_generator: Iterator[TransformerGraphBundleInput], model: EncoderDecoder, loss_compute: SimpleLossCompute):
    "Standard Training and Logging Function"
    start = time.time()
    total_loss = 0
    ntokens = 0
    for subgraph_bundle in subgraph_bundle_generator: 
        out = model(subgraph_bundle.src_feats, subgraph_bundle.src_mask,  
                            subgraph_bundle.train_inds) # B x B_out x model_D.  
        # TODO: need to think about this loss computation carefully. Is it even possible?
        total_loss += loss_compute(out, subgraph_bundle.trg_labels, subgraph_bundle.ntokens)
        ntokens += subgraph_bundle.ntokens 
    elapsed = time.time() - start
    print(f"Train loss on epoch: {total_loss / ntokens}; time taken: {elapsed}")
    return total_loss / ntokens, elapsed

def run_eval_epoch(subgraph_bundle_generator: Iterator[TransformerGraphBundleInput], model: EncoderDecoder, \
                    loss_compute: SimpleLossCompute):
    start = time.time()
    total_loss = 0
    ntokens = 0
    for subgraph_bundle in subgraph_bundle_generator: 
        out = model(subgraph_bundle.src_feats, subgraph_bundle.src_mask,  
                            subgraph_bundle.train_inds) # B x B_out x model_D.  
        # TODO: need to think about this loss computation carefully. Is it even possible?
        total_loss += loss_compute(out, subgraph_bundle.trg_labels, subgraph_bundle.ntokens)
        ntokens += subgraph_bundle.ntokens 
    elapsed = time.time() - start
    print(f"Validation loss on epoch: {total_loss / ntokens}")
    return total_loss / ntokens 

def eval_accuracy(graph_bundle: TransformerGraphBundleInput, model: EncoderDecoder):
    total = 0
    num_correct = 0 
    out = model(graph_bundle.src_feats, graph_bundle.src_mask, graph_bundle.train_inds) # B x B_out x model_D.  
    out = model.generator(out) # B x num_nodes x num_classes
    out = out.squeeze(0) # num_nodes x num_classes
    out = out.argmax(axis=1) # num_nodes
    mb_test_labels = graph_bundle.trg_labels.squeeze(0)
    total += mb_test_labels.shape[0]
    num_correct += (out == mb_test_labels).sum()
    test_accuracy = num_correct / total
    print(test_accuracy)
    return test_accuracy

def eval_accuracy_mb(subgraph_bundle_generator: Iterator[TransformerGraphBundleInput], model: EncoderDecoder):
    total = 0
    num_correct = 0 
    predictions = []
    test_labels = []
    for graph_bundle in subgraph_bundle_generator:
        out = model(graph_bundle.src_feats, graph_bundle.src_mask, graph_bundle.train_inds)
        out = model.generator(out) # B x num_nodes x num_classes
        out = out.squeeze(0) # num_nodes x num_classes
        out = out.argmax(axis=1) # num_nodes
        predictions.append(out)
        mb_test_labels = graph_bundle.trg_labels.squeeze(0)
        test_labels.append(mb_test_labels)
    predictions = torch.cat(predictions)
    test_labels = torch.cat(test_labels)
    num_correct = sum(predictions == test_labels)
    total = len(test_labels)
    test_accuracy = num_correct / total
    print(test_accuracy)
    return test_accuracy


def get_input_output_dims():
    data = citegrh.load_cora()
    return data.features.shape[1], len(torch.tensor(data.labels).unique())

def instantiate_model(input_dim, output_num_classes, hparams_d):
    """Instantiate T-GNN. 

    Args:
        hparams_d (): hyperparameters from search.
    """

# def train_model(input_dim, output_dim, features, train_nids, val_nids, graph, labels, gpu, hparams_d):
def train_model(hparams_d):
    """Train the GraphTransformer model for 32 epochs.

    Args:
        bs (int): batch size
        num_sg (int): number of subgraphs. If 1, then no padding is done.
    """
    gpu = 0
    input_dim, output_num_classes = 1433, 7
    data = citegrh.load_cora()
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    graph = data[0]
    adj = graph.adj(scipy_fmt='coo')

    features = data.features.clone().detach().to(0)
    labels = torch.tensor(data.labels, device=(0)) 
    graph = dgl.graph((adj.row, adj.col)).to(0)
    train_nids = (torch.arange(0, graph.number_of_nodes())[train_mask]).to(0)
    val_nids = (torch.arange(0, graph.number_of_nodes())[val_mask]).to(0)
    test_nids = (torch.arange(0, graph.number_of_nodes())[test_mask]).to(0)

    num_heads = 2 ** int(hparams_d['num_heads']) # TODO: quniform
    head_dim  = 2 ** int(hparams_d['head_dim']) # TODO: quniform
    d_model = num_heads * head_dim 
    d_ff = 2 ** int(hparams_d['d_ff']) # TODO: quniform
    dropout = hparams_d['dropout'] # TODO: log normal

    model = make_model(input_dim, output_num_classes, N=2, d_model=d_model, d_ff=d_ff, h=num_heads, dropout=dropout).to(gpu)

    print(f"Training with : {hparams_d}")
    device = gpu

    # model = make_model(input_dim, output_num_classes, N=2, d_model=args.d_model, d_ff=args.d_ff, dropout=0.6).to(0) # +1 for the padding index, though i don't think it's necessary.
    

    fanout_inner = int(hparams_d['fanout_0']) # quniform
    fanout_outer = fanout_inner * hparams_d['fanout_1_scale_factor'] # choice: 1,2,3

    batch_size = hparams_d['batch_size'] # choice
    criterion = LabelSmoothing(size=7, smoothing=hparams_d['label_smoothing']).to(device) # loguniform
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, hparams_d['warmup'], # quniform... dont have much expectations over this.
        torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-6, weight_decay=hparams_d['weight_decay'])) # qloguniform?
    nepochs = 10
    best_loss = float("inf") 

    sampler = dgl.dataloading.MultiLayerNeighborSampler([fanout_inner, fanout_outer])
    train_dataloader = dgl.dataloading.DataLoader(
        graph, train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0 
        )
    val_dataloader = dgl.dataloading.DataLoader(
        graph, val_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0 
        )
    num_subgraphs = 1
    best_loss_epoch = 0

    tb_log_dir = f"runs/batch-{batch_size}_num_sg-{num_subgraphs}"
    tb_sw = SummaryWriter(tb_log_dir)
    for nepoch in range(nepochs):
        model.train()
        nbatches = train_nids.shape[0] // batch_size
        epoch_loss, train_epoch_elapsed  = run_train_epoch(cora_data_gen(train_dataloader, nbatches, num_subgraphs, features, labels, device), model, 
            SimpleLossCompute(model.generator, criterion, model_opt))
        
        tb_sw.add_scalar('Loss/train', epoch_loss, nepoch)
        tb_sw.add_scalar('Duration/train', train_epoch_elapsed, nepoch)
        
        model.eval()
        with torch.no_grad():
            validation_loss = run_eval_epoch(cora_data_gen(val_dataloader, nbatches, 1, features, labels, device), model, 
                SimpleLossCompute(model.generator, criterion, None))
            tb_sw.add_scalar('Loss/validation', validation_loss, nepoch)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_loss_epoch = nepoch
    best_val_loss =best_loss.item()
    with torch.no_grad():
        del model
        features = None
        labels = None
        graph = None
        train_nids = None 
        val_nids = None
        sampler = None
        train_dataloader = None
        val_dataloader = None
        criterion = None
        torch.cuda.empty_cache()
        gc.collect()
    return best_val_loss


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main_proc(rank, world_size):
    setup(rank, world_size)
    input_dim, output_num_classes = get_input_output_dims() 
    model = make_model(input_dim, output_num_classes + 1, N=2).to(rank) # +1 for the padding index, though i don't think it's necessary.
    model = ddp(model, device_ids=[rank], output_device=rank, find_unused_parameters=false)

    # TODO: call train_model
    train_model(model, rank)

def build_dataloader(graph: dgl.DGLHeteroGraph, bs: int, fanouts: List[int], ids: torch.tensor):
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.DataLoader(
        graph, ids, sampler,
        batch_size=bs,
        shuffle=True,
        drop_last=False
    )
    return dataloader


def evaluate_model(model, gpu, batch_size, fanout_inner, fanout_outer):
    data = citegrh.load_cora()
    bs = batch_size
    features = data.features.clone().detach().to(gpu)
    labels = torch.tensor(data.labels, device=(gpu)) 
    test_mask = torch.BoolTensor(data.test_mask)
    graph = data[0]
    adj = graph.adj(scipy_fmt='coo')
    adj_tens = torch.tensor(adj.todense(), device=gpu) 
    graph = dgl.graph((adj.row, adj.col)).to(gpu)

    test_nids = (torch.arange(0, graph.number_of_nodes())[test_mask]).to(gpu)
    test_dataloader = build_dataloader(graph, 32, [fanout_inner, fanout_outer], test_nids)
    input_dim, output_num_classes = get_input_output_dims() 
    load_model(model)
    model.eval()
    test_nbatches = test_nids.shape[0] // bs
    with torch.no_grad():
        # test_acc = eval_accuracy_mb(cora_data_gen(test_dataloader, test_nbatches, 1, features, labels, gpu), model)
        test_acc = eval_accuracy(test_cora_data_gen(adj_tens, features, test_nids, labels, gpu), model)
    print(f"{test_acc:.3f}")

def main_global(args):
    if args.evaluate_model:
        evaluate_model(0)
    else:
        # train_model(model, 0, args.batch_size, args.fanout_inner, args.fanout_outer, 0.052)

        # space = hp.uniform('a', 0, 0.20)

        # TODO: be careful with the log.
        space = {
            'num_heads': hp.quniform('num_heads', 1, 5, 1),
            'head_dim':  hp.quniform('head_dim', 4, 7, 1),
            'd_ff':  hp.quniform('d_ff', 5, 11, 1),
            'dropout': hp.loguniform('dropout', np.log(0.1), np.log(0.8)),
            'fanout_0': hp.quniform('fanout_0', 5, 35, 1),
            'fanout_1_scale_factor': hp.choice('fanout_1_scale_factor', [1,2,3]),
            'label_smoothing': hp.loguniform('label_smoothing', np.log(0.01), np.log(0.7)),
            'warmup': hp.quniform('warmup', 25, 400, 1),
            'weight_decay': hp.loguniform('weight_decay', np.log(1e-4), np.log(1e-2)),
            'batch_size': hp.choice('batch_size', [32, 64])

        }
        # objective = partial(train_model, model, 0, args.batch_size, args.fanout_inner, args.fanout_outer)
        objective = train_model
        best = fmin(objective, space, algo=tpe.suggest, max_evals=5)
        print(best)

        # results.get_dataframe().to_csv("results/ray_tune_first_run.csv")
        # evaluate_model(model, 0, args.batch_size, args.fanout_inner, args.fanout_outer)

    # world_size = args.num_gpus
    # mp.spawn(main_proc, args=(world_size, ), nprocs = world_size, join=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("bs", type=int)
    # parser.add_argument("num_sg", type=int)
    parser.add_argument("--evaluate_model", action='store_true')
    # parser.add_argument("d_model", type=int)
    # parser.add_argument("d_ff", type=int)
    # parser.add_argument("batch_size", type=int)
    # parser.add_argument("fanout_inner", type=int)
    # parser.add_argument("fanout_outer", type=int)
    main_global(parser.parse_args())