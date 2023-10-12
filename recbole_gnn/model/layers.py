import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import MessagePassing, HypergraphConv, DMoNPooling, GCNConv, DenseGraphConv, GraphConv
from torch_geometric.utils import add_self_loops, to_dense_adj

import faiss
from typing import Optional


class TimeEncode(nn.Module):
    """
    https://github.com/CongWeilin/GraphMixer
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.alpha = math.sqrt(dim)
        self.beta = math.sqrt(dim)
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self, ):
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / self.alpha ** np.linspace(0, self.beta - 1, self.dim, dtype=np.float32))).reshape(
                self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        if len(t.shape) > 1:
            return output.reshape(-1, t.shape[1], self.dim)
        return output


class SimpleHypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None) -> Tensor:
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             size=(num_edges, num_nodes))

        out = out.view(-1, self.out_channels)

        return out

    def message(self, x_j: Tensor, norm_i: Tensor) -> Tensor:
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, 1, self.out_channels)

        return out

class GNN(nn.Module):
    def __init__(self, hidden_size, n_layers=2, dropout=0.2):
        super(GNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # normalize = False, needed to be deterministic?
        self.convs = nn.ModuleList([GCNConv(hidden_size, hidden_size, cached=True, normalize=False, add_self_loops=True)
                                    ] * n_layers)
        self.convs = nn.ModuleList([GraphConv(hidden_size, hidden_size)
                                    ] * n_layers)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_p = dropout

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        final = [x]
        for i in range(self.n_layers):
            h = self.convs[i](h, edge_index, edge_weight)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
            final.append(h)

        if self.n_layers > 1:
            h = torch.sum(torch.stack(final), dim=0) / (self.n_layers + 1)

        return h

class HGNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super(HGNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.convs = nn.ModuleList([SimpleHypergraphConv(hidden_size, hidden_size)
                                    ] * n_layers)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        final = [x]
        for i in range(self.n_layers):
            h = self.convs[i](h, edge_index, hyperedge_weight=edge_weight)
            final.append(h)

        h = torch.sum(torch.stack(final), dim=0) / (self.n_layers + 1)

        return h


class HClusterGNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, n_clusters=512, dropout=0.2):
        super(HClusterGNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_clusters = n_clusters

        self.hyper_convs = nn.ModuleList([HypergraphConv(hidden_size, hidden_size)
                                          ] * n_layers)
        #self.convs = nn.ModuleList([GCNConv(hidden_size, hidden_size, cached=True, normalize=True)
        #                                  ] * 2)
        self.convs = nn.ModuleList([GraphConv(hidden_size, hidden_size)
                                    ] * n_layers)

        self.pool = DMoNPooling([self.hidden_size] * n_layers, self.n_clusters, dropout=dropout)

        self.dropout_p = dropout
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, hyper_edge_index, c_edge_index, hyper_edge_weight=None, c_edge_weight=None):
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.convs[0](x, c_edge_index, c_edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.convs[1](x, c_edge_index, c_edge_weight)

        '''x_l = [F.normalize(x, dim=-1, p=2)]
        for i in range(self.n_layers):
            x = self.hyper_convs[i](x, hyper_edge_index, hyperedge_weight=hyper_edge_weight)
            x = F.normalize(x, dim=-1, p=2) # TODO with LayerNorm, same?
            #x = self.activation(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            x_l.append(x)

        if self.n_layers > 1:
            x = torch.sum(torch.stack(x_l), dim=0) / (self.n_layers + 1)'''

        adj = to_dense_adj(c_edge_index, edge_attr=c_edge_weight)
        cluster_prob, cluster_emb, adj, sp_l, o_l, c_l = self.pool(x, adj)

        cluster_loss = sp_l + o_l + c_l
        cluster_emb = cluster_emb.squeeze()

        node2cluster = cluster_prob.squeeze().argmax(dim=1)
        #print(len(torch.unique(node2cluster)))

        return x, node2cluster, cluster_emb, cluster_loss


class SimilarityFunction(nn.Module):
    def __init__(self, sim_name, hidden_size):
        super(SimilarityFunction, self).__init__()
        self.sim_name = sim_name
        self.hidden_size = hidden_size

        self.metric_w1 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.metric_w2 = nn.Parameter(torch.Tensor(1, self.hidden_size))

    def metric_sim(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = torch.stack((self.metric_w1 * x1, self.metric_w2 * x1)).mean(0)
        x2 = torch.stack((self.metric_w1 * x2,
                          self.metric_w2 * x2)).mean(0).unsqueeze(1)

        # cosine similarity estimation
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        return torch.matmul(x1, x2.permute(0, 2, 1)).squeeze()

    def cosine_dot_sim(self, x1: torch.Tensor, x2: torch.Tensor, dim=2):
        x2 = x2.unsqueeze(1)
        # https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity?hl=en
        return torch.sum(torch.mul(x1, x2), dim=dim)  # dot product, cosine operation

    def cosine_sim(self, x1: torch.Tensor, x2: torch.Tensor, dim=2):
        x2 = x2.unsqueeze(1)
        return torch.cosine_similarity(x1, x2, dim=dim)

    def euclidean_sim(self, x1: torch.Tensor, x2: torch.Tensor, dim=2):
        x2 = x2.unsqueeze(1)
        return 1 - (x1 - x2).pow(2).sum(dim=dim).sqrt()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        if self.sim_name == 'metric_learning':
            return self.metric_sim(x1, x2)
        elif self.sim_name == 'cosine':
            return self.cosine_sim(x1, x2)
        elif self.sim_name == 'cosine_dot':
            return self.cosine_dot_sim(x1, x2)
        elif self.sim_name == 'euclidean':
            return self.euclidean_sim(x1, x2)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.sim_name)


class FusionAttention(nn.Module):
    """
    https://github.com/wangjlgz/HyperRec/blob/master/model.py
    """

    def __init__(self, hidden_size):
        super(FusionAttention, self).__init__()
        self.hidden_size = hidden_size

        self.w_f = nn.Linear(self.hidden_size, self.hidden_size)
        self.z = nn.Linear(self.hidden_size, 1)

    def forward(self, item_intensity_emb, seq_output):
        stacked_input = torch.stack([item_intensity_emb, seq_output], dim=1)
        nh = self.w_f(stacked_input)
        nh = torch.tanh(nh)
        beta = self.z(nh)
        beta = torch.softmax(beta, dim=1)

        fused_input = torch.sum(beta * stacked_input, 1)

        return fused_input


class SimpleScoreFusionLayer(nn.Module):
    def __init__(self, hidden_size, sub_time_delta):
        super(SimpleScoreFusionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.sub_time_delta = sub_time_delta
        # self.time_encoder = TimeEncode(self.hidden_size)

    def forward(self, item_seq, item_emb, target_item, item_intensity, item_score, time, time_seq):
        time_len = time_seq.gt(0).sum(dim=1)
        last_inter_time = time_seq.gather(dim=1, index=time_len.unsqueeze(-1) - 1).squeeze()
        time_delta = time - last_inter_time
        time_delta_mask = (time_delta > self.sub_time_delta)

        fused_score = (~time_delta_mask * item_score) + (time_delta_mask * item_intensity.squeeze())

        return fused_score


class ScoreFusionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(ScoreFusionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.time_encoder = TimeEncode(self.hidden_size)
        self.w = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, item_intensity, item_score, time, time_seq):
        item_score = item_score.unsqueeze(-1)

        seq_len = time_seq.gt(0).sum(dim=1)
        last_inter_time = time_seq.gather(dim=1, index=seq_len.unsqueeze(-1) - 1).squeeze()

        last_inter_time_emb = self.time_encoder(last_inter_time)
        time_emb = self.time_encoder(time)

        concat_emb = torch.cat([last_inter_time_emb, time_emb], dim=1)

        w_combine = torch.sigmoid(self.w(concat_emb))
        fused_score = w_combine * item_score + ((1 - w_combine) * item_intensity)

        return fused_score.squeeze()


class AttentionPositionReadout(nn.Module):
    def __init__(self, hidden_size, max_seq_length):
        super(AttentionPositionReadout, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length

        # last subseq attention + pos
        self.pos_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.w_1 = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.w_2 = nn.Linear(self.hidden_size, 1, bias=False)
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, seq, seq_hidden):
        batch_size = seq.shape[0]
        len = seq.shape[1]
        mask = seq.gt(0).unsqueeze(-1)

        hs = torch.sum(seq_hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)

        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = torch.flip(pos_emb, [0])  # reverse order
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_hidden], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))

        beta = self.w_2(nh)
        beta = beta * mask
        seq_output = torch.sum(beta * seq_hidden, 1)

        return seq_output


class KMeans(object):
    """
    https://github.com/salesforce/ICLRec
    https://github.com/facebookresearch/faiss/blob/main/benchs/kmeans_mnist.py
    """

    def __init__(self, num_cluster, seed, hidden_size, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        # self.gpu_id = 0
        self.device = device
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []
        self.trainable = True

    def __init_cluster(
            self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()  # use a single GPU
        index_flat = faiss.IndexFlatL2(self.hidden_size)  # build a flat (CPU) index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        return clus, gpu_index_flat

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x=x.cpu().detach().numpy(), index=self.index)
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        centroids = torch.Tensor(centroids).to(self.device)
        # self.centroids = nn.functional.normalize(centroids, p=2, dim=1)
        self.centroids = centroids

    def query(self, x):
        D, I = self.index.search(x.cpu().detach().numpy(), 1)  # for each sample, find cluster distance and assignments
        x2cluster = [int(n[0]) for n in I]
        x2cluster = torch.LongTensor(x2cluster).to(self.device)
        return x2cluster, self.centroids[x2cluster]


class LBA(nn.Module):
    """Location-based attention layer. FISSA"""

    def __init__(self, dim, dropout=0.2):
        super(LBA, self).__init__()
        self.dropout = dropout
        self.dim = dim

        self.W = nn.Linear(dim, dim)  # att weights
        self.W_2 = nn.Linear(dim, dim)  # att weights 2
        self.q = nn.Parameter(torch.Tensor(1, self.dim))  # query weight

    def forward(self, k, v, mask):
        k = self.W(k)  # [B, seq_len, dim]
        qk = torch.matmul(k, self.q.T)  # [B, seq_len, 1]
        v = self.W_2(v)  # [B, seq_len, dim]

        attention_scores = qk / math.sqrt(self.dim)

        mask = torch.where(mask, 0.0, -10000.0)  # for stable softmax

        attention_scores = attention_scores + mask.unsqueeze(-1)  # [B, seq_len, 1]
        attention_probs = torch.softmax(attention_scores, dim=1).transpose(2, 1)  # [B, 1, seq_len]

        outputs = torch.matmul(attention_probs, v).squeeze(1)  # [B, dim]
        outputs = F.dropout(outputs, p=self.dropout, training=self.training)
        return outputs


class AttentionMixer(nn.Module):
    def __init__(self, hidden_size, levels=7, n_heads=2, dropout=0.2):
        super(AttentionMixer, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.n_levels = levels
        self.dropout = dropout

        self.level_queries = nn.ModuleList([nn.Linear(hidden_size, hidden_size)] * self.n_levels)

        self.hidden_size = hidden_size
        self.n_heads = n_heads

        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.lp_pool = nn.LPPool1d(4, kernel_size=self.n_levels, stride=self.n_levels)
    

    def forward(self, item_seq, item_seq_emb, item_seq_len):
        mask = item_seq.gt(0)

        # generate deep sets - multi query generation
        queries = []
        for i in range(self.n_levels):
            seq_ids = torch.arange(mask.size(0)).long()
            level_emb = [item_seq_emb[seq_ids, torch.clamp(item_seq_len - (j + 1), -1, 1000)] for j in range(i + 1)]
            level_emb = torch.sum(torch.stack(level_emb, dim=1), dim=1)
            queries.append(self.level_queries[i](level_emb).unsqueeze(1))

        queries = torch.stack(queries, dim=1)

        query_layer = self.query(queries).view(-1, queries.size(1), self.hidden_size // self.n_heads)
        key_layer = self.key(item_seq_emb).view(-1, item_seq_emb.size(1),
                                          self.hidden_size // self.n_heads)  # batch_size x seq_length x latent_size
        value = item_seq_emb.view(-1, item_seq_emb.size(1), self.hidden_size // self.n_heads)

        alpha = torch.sigmoid(torch.matmul(query_layer, key_layer.permute(0, 2, 1)))
        alpha = alpha.view(-1, query_layer.size(1) * self.n_heads, item_seq_emb.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(alpha, dim=1)

        alpha = self.lp_pool(alpha)
        alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
        alpha = torch.softmax(alpha, dim=1)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        output = torch.sum(
            (alpha.unsqueeze(-1) * value.view(item_seq_emb.size(0), -1, self.n_heads, self.hidden_size // self.n_heads)).view(
                item_seq_emb.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)

        return output


class LastAttention(nn.Module):
    """
    https://github.com/Peiyance/Atten-Mixer-torch
    """
    def __init__(self, hidden_size, levels=10, n_heads=2, dropout=0.2):
        super(LastAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.hidden_size = hidden_size
        self.dropout = dropout

        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.hidden_size = hidden_size
        self.n_heads = n_heads

        self.lp_pool = nn.LPPool1d(4, kernel_size=levels, stride=levels)
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):
        q0 = self.linear_zero(ht1).view(-1, ht1.size(1), self.hidden_size // self.n_heads)
        q1 = self.linear_one(hidden).view(-1, hidden.size(1),
                                          self.hidden_size // self.n_heads)  # batch_size x seq_length x latent_size
        q2 = self.linear_two(hidden).view(-1, hidden.size(1), self.hidden_size // self.n_heads)
        alpha = torch.sigmoid(torch.matmul(q0, q1.permute(0, 2, 1)))
        alpha = alpha.view(-1, q0.size(1) * self.n_heads, hidden.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(2 * alpha, dim=1)

        alpha = self.lp_pool(alpha)
        alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
        alpha = torch.softmax(2 * alpha, dim=1)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        attn = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.n_heads, self.hidden_size // self.n_heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        attn = self.last_layernorm(attn)

        return attn


class LightGCNConv(MessagePassing):
    def __init__(self, dim):
        super(LightGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BipartiteGCNConv(MessagePassing):
    def __init__(self, dim):
        super(BipartiteGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight, size):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BiGNNConv(MessagePassing):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin1 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.lin2 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_weight):
        x_prop = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x_trans = self.lin1(x_prop + x)
        x_inter = self.lin2(torch.mul(x_prop, x))
        return x_trans + x_inter

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        # mean aggregation to incorporate weight naturally
        super(SRGNNConv, self).__init__(aggr='mean')

        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)


class SRGNNCell(nn.Module):
    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.dim = dim
        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

        self._reset_parameters()

    def forward(self, hidden, edge_index):
        input_in = self.incomming_conv(hidden, edge_index)
        reversed_edge_index = torch.flip(edge_index, dims=[0])
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
