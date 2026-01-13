#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from data.drugs import smiles_to_graph

class MultimodalDataset(Dataset):
    """
    cnn_data: (N, 1, H, W) 텐서 또는 Lazy Tensor(인덱싱 가능)
    smiles_data: 길이 N의 시퀀스(리스트/시리즈) – 각 원소는 SMILES 문자열
    labels: (N,) 텐서
    """
    def __init__(self, cnn_data, smiles_data, labels):
        self.cnn_data = cnn_data
        self.smiles_data = smiles_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cnn_input = self.cnn_data[idx]
        smiles = self.smiles_data[idx]
        label = self.labels[idx]
        gcn_data = smiles_to_graph(smiles)
        return cnn_input, gcn_data, label

def collate_fn(batch):
    cnn_inputs, gcn_data, labels = zip(*batch)
    cnn_inputs = torch.stack(cnn_inputs)
    gcn_batch = Batch.from_data_list(gcn_data)
    labels = torch.stack(labels)
    return cnn_inputs, gcn_batch, labels

