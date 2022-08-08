from argparse import ArgumentParser
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

def get_input_output_dims():
    data = citegrh.load_cora()
    return data.features.shape[1], len(torch.tensor(data.labels).unique())

def train_model(model, gpu):
    """Train the GraphTransformer model for 32 epochs.

    Args:
        bs (int): batch size
        num_sg (int): number of subgraphs. If 1, then no padding is done.
    """

    data = citegrh.load_cora()
    features = data.features.clone().detach().to(gpu)
    labels = torch.tensor(data.labels, device=(gpu)) 
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    graph = data[0]
    adj = graph.adj(scipy_fmt='coo')
    graph = dgl.graph((adj.row, adj.col)).to(gpu)
    device = gpu

    criterion = LabelSmoothing(size=8, padding_idx=7, smoothing=0.0).to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-6))
    nepochs = 100
    best_loss = float("inf") 
    train_nids = (torch.arange(0, graph.number_of_nodes())[train_mask]).to(device)
    val_nids = (torch.arange(0, graph.number_of_nodes())[val_mask]).to(device)
    test_nids = (torch.arange(0, graph.number_of_nodes())[test_mask]).to(device)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 5])
    batch_size = 32
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
    num_subgraphs = 2
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
                checkpoint_model(model)
                best_loss = validation_loss
                best_loss_epoch = nepoch
            
    print(f"Best validation epoch: {best_loss_epoch}")
    load_model(model) # mutation
    model.eval()
    with torch.no_grad():
        test_labels = labels[test_nids]
        test_acc = eval_accuracy(test_cora_data_gen(graph.adj().to_dense().to(device) + torch.eye(adj.shape[0]).to(device), features, test_nids, test_labels, device), model)
        tb_sw.add_scalar('Accuracy/test', test_acc)
    print(f"{test_acc:.3f},{best_loss:.3f},{best_loss_epoch}")

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main_proc_ddp(rank, world_size):
    setup(rank, world_size)
    input_dim, output_num_classes = get_input_output_dims() 
    model = make_model(input_dim, output_num_classes + 1, N=2).to(rank) # +1 for the padding index, though I don't think it's necessary.
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # TODO: call train_model
    train_model(model, rank)

def main_global(args):
    if args.use_ddp:
        world_size = args.num_gpus
        mp.spawn(main_proc_ddp, args=(world_size, ), nprocs = world_size, join=True)
    else:
        print("Not using DDP")
        input_dim, output_num_classes = get_input_output_dims() 
        model = make_model(input_dim, output_num_classes + 1, N=2).to(0) # +1 for the padding index, though I don't think it's necessary.
        train_model(model, 0)

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("bs", type=int)
    # parser.add_argument("num_sg", type=int)
    parser.add_argument("--use_ddp", action='store_true')
    parser.add_argument("num_gpus", type=int)
    main_global(parser.parse_args())