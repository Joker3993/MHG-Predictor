import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init
import dgl.nn as dglnn
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv, GraphConv, GATv2Conv, SAGEConv, GatedGraphConv


class HeteroAttentionLayer(nn.Module):
    def __init__(self, in_size, out_size, rel_names, node_name):
        super(HeteroAttentionLayer, self).__init__()
        self.node_name = node_name
        self.conv1 = dglnn.HeteroGraphConv({
            rel: SAGEConv(in_feats=in_size, out_feats=out_size, aggregator_type="lstm", feat_drop=0.3)
            for rel in rel_names})
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.ModuleDict({rel: nn.LayerNorm(out_size) for rel in node_name})
        self.norm3 = nn.ModuleDict({rel: nn.LayerNorm(out_size) for rel in node_name})
        self.reset_parameters()

    def reset_parameters(self):
        for conv in [self.conv1]:
            for _, submodule in conv.named_modules():
                if isinstance(submodule, nn.Linear):
                    init.kaiming_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        init.zeros_(submodule.bias)

    def forward(self, g, inputs):
        list = []
        h_residual = inputs
        h = self.conv1(g, inputs)
        for k, v in h.items():
            h[k] = self.norm1[k](h[k])
            h[k] = F.leaky_relu(h[k])
        h = {**inputs, **h}
        for node_name in self.node_name:
            h[node_name] = h_residual[node_name] + h[node_name]
            h[node_name] = self.norm3[node_name](h[node_name])
            h[node_name] = F.leaky_relu(h[node_name])
            h[node_name] = self.dropout(h[node_name])
        return h


class HomoLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(HomoLayer, self).__init__()
        self.conv1 = GatedGraphConv(in_feats=hidden_dim, out_feats=hidden_dim, n_steps=2, n_etypes=1)
        self.homo_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        for conv in [self.conv1]:
            for param in conv.parameters():
                if param.dim() > 1:
                    init.kaiming_uniform_(param)
            if hasattr(conv, 'bias') and conv.bias is not None:
                init.zeros_(conv.bias)
        for norm in [self.homo_norm1]:
            for param in norm.parameters():
                if param.dim() > 1:
                    init.kaiming_uniform_(param)
            if hasattr(norm, 'bias') and norm.bias is not None:
                init.zeros_(norm.bias)

    def forward(self, g2, h_g2):
        h_residual1 = h_g2
        h_g2 = self.conv1(g2, h_g2)
        h_g2 = self.homo_norm1(h_g2 + h_residual1)
        h_g2 = F.leaky_relu(h_g2)
        h_g2 = self.dropout1(h_g2)
        return h_g2


class SAGE_Classifier(nn.Module):
    def __init__(self, hidden_dim, n_classes, num_heads, num_layers, rel_names, vocab_sizes, node_name):
        super(SAGE_Classifier, self).__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(voca_size, hidden_dim) for voca_size in vocab_sizes
        ])
        self.node_name = node_name
        self.HeteroAttentionLayer = nn.ModuleList(
            [HeteroAttentionLayer(hidden_dim, hidden_dim, rel_names, node_name) for _ in range(num_layers)])
        self.norm1 = nn.ModuleDict({node_name: nn.LayerNorm(hidden_dim) for node_name in node_name})
        self.homoLayer = nn.ModuleList(
            [HomoLayer(hidden_dim) for _ in range(1)])
        self.transform = nn.Linear(2 * hidden_dim, hidden_dim)
        self.MLP_hetro = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.MLP_homo = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.classify = nn.Linear(hidden_dim, n_classes)
        for emb in self.embedding_layers:
            init.kaiming_uniform_(emb.weight)
        self.embedding_activity2 = nn.Embedding(30, hidden_dim)
        init.kaiming_uniform_(self.embedding_activity2.weight)
        for norm in self.norm1.values():
            for param in norm.parameters():
                if param.dim() > 1:
                    init.kaiming_uniform_(param)
            if hasattr(norm, 'bias') and norm.bias is not None:
                init.zeros_(norm.bias)
        for param in self.classify.parameters():
            if param.dim() > 1:
                init.kaiming_uniform_(param)
        if self.classify.bias is not None:
            init.zeros_(self.classify.bias)
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Embedding):
            init.kaiming_uniform_(m.weight.data)
        elif isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)

    def forward(self, g, g2):
        h = {}
        for node_name, embedding in zip(self.node_name, self.embedding_layers):
            h_feature = embedding(g.nodes[node_name].data[node_name].type(torch.long))
            h[node_name] = h_feature
        hg2_list = []
        h_g2 = self.embedding_activity2(g2.ndata['activity'].type(torch.long))
        h_combined = torch.cat((h_g2, h['duration']), dim=1)
        h_new = self.transform(h_combined)
        for homoLayer in self.homoLayer:
            h_new = homoLayer(g2, h_new)

        with g2.local_scope():
            g2.ndata['h'] = h_new
            hg2_mean = dgl.mean_nodes(g2, 'h')

        hg_list = []
        h_gat = h
        for hetroLayer in self.HeteroAttentionLayer:
            h_gat = hetroLayer(g, h_gat)
        with g.local_scope():
            g.ndata['h'] = h_gat
            hg_max = 0
            hg_mean = 0

            for ntype in g.ntypes:
                hg_mean = hg_mean + dgl.mean_nodes(g, 'h', ntype=ntype)

        output1 = self.MLP_homo(hg2_mean)
        output2 = self.MLP_hetro(hg_mean)
        hg_combined = torch.cat((
            output1, hg2_mean, output2, hg_mean
        ), dim=1)
        preds = self.MLP(hg_combined)
        return self.classify(preds)
