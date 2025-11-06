#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ======================================
# 1. CNN 모델 (Variant Encoder)
# ======================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(20, 7))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(20, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        self.fc1 = nn.Linear(34400, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)      # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# ======================================
# 2. GCN 모델 (Drug Encoder)
# ======================================
class GCNModel(nn.Module):
    def __init__(self, num_node_features, hidden_channels=128):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.fc = nn.Linear(hidden_channels, 128)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc(x))
        return x


# ======================================
# 3. Multimodal Fusion Model
# ======================================
class MultimodalModel(nn.Module):
    def __init__(self, cnn_model, gcn_model,
                 cnn_output_dim=128, gcn_output_dim=128, num_classes=3):
        super(MultimodalModel, self).__init__()
        self.cnn_model = cnn_model
        self.gcn_model = gcn_model
        self.fc1 = nn.Linear(cnn_output_dim + gcn_output_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, cnn_input, gcn_data):
        cnn_output = self.cnn_model(cnn_input)
        gcn_output = self.gcn_model(gcn_data)
        x = torch.cat([cnn_output, gcn_output], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

