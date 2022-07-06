import dgl
from numpy import dtype
import torch
from dgl.data import citation_graph as citegrh
import pytest
from packages.transformer.data import load_cora_data, retrieve_features_for_minibatch, retrieve_labels_for_minibatch

@pytest.fixture
def cora_dataloader():
    data = citegrh.load_cora()
    train_mask = torch.BoolTensor(data.train_mask)
    graph = data[0]
    adj = graph.adj(scipy_fmt='coo')
    graph = dgl.graph((adj.row, adj.col))
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 5])
    train_nids = (torch.arange(0, graph.number_of_nodes())[train_mask])
    dataloader = dgl.dataloading.DataLoader(
        graph, train_nids, sampler,
        batch_size=20,
        shuffle=True,
        drop_last=False,
        num_workers=0)
    dataloader_iter = iter(dataloader)
    return dataloader_iter

@pytest.fixture
def feature_matrix():
    feats = torch.ones(10, 1433) * torch.arange(1,11)[:,None]
    return feats

def test_argsort_inds(cora_dataloader):
    input_nodes, output_nodes, mfgs = next(cora_dataloader) # input nodes gives us the requisite features. The mfgs gives us the requisite attention mask
    src_inds_first_layer = (mfgs[0].srcdata[dgl.NID])
    all_input_argsort_inds_src_inds = torch.argsort(src_inds_first_layer)

    all_input_argsort_inds_input_nodes = torch.argsort(input_nodes)
    assert torch.equal(all_input_argsort_inds_input_nodes, all_input_argsort_inds_src_inds)

def test_retrieve_feats_for_minibatch(feature_matrix):
    global_inds = torch.Tensor([9,8,1,4,2]).long()
    argsort_inds = torch.argsort(global_inds)
    actual = retrieve_features_for_minibatch(global_inds, argsort_inds, feature_matrix)
    expected = ((feature_matrix[[1,2,4,8,9],:]))
    assert torch.equal(actual, expected)

def test_retrieve_labels_for_minibatch():
    
