# @Time   : 2023/3/15
# @Author : Andreas Peintner
# @Email  : a.peintner@gmx.net

r"""
HyperHawkes
################################################

Reference:
    TODO

"""
import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn.init import xavier_normal_, constant_
from torch_geometric.nn import DMoNPooling, GraphMultisetTransformer, TopKPooling, ASAPooling
from tqdm import tqdm

from recbole.utils import InputType
from recbole_custom.model.loss import BPRLoss, EmbLoss, BCELoss
from recbole.model.abstract_recommender import GeneralRecommender, SequentialRecommender
from recbole.model.init import xavier_normal_initialization, xavier_uniform_initialization

from recbole_gnn.data.dataset import GeneralGraphDataset
from recbole_gnn.model.layers import HClusterGNN, SimilarityFunction, TimeEncode, AttentionMixer, \
    AttentionPositionReadout, HGNN, \
    ScoreFusionLayer, SimpleScoreFusionLayer, LBA, KMeans, GNN

import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain, combinations
from torch_geometric.utils import degree, k_hop_subgraph, add_self_loops, to_torch_coo_tensor

import faiss


class HyperHawkes(SequentialRecommender):
    r"""
    HyperHawkes models user sequences via a global hyper graph and an intent-based hawkes process for recommendation.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HyperHawkes, self).__init__(config, dataset)

        # load parameters info
        self.config = config
        self.dataset = dataset
        self.hidden_size = config["hidden_size"]
        self.hgnn_layers = config['hgnn_layers']
        self.n_levels = config["n_levels"]
        self.n_heads = config["n_heads"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.emb_dropout_prob = config["emb_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.sub_time_delta = config['sub_time_delta']
        self.day_factor = config['day_factor']
        self.n_clusters = config['n_clusters']

        self.use_atten_mixer = config['use_atten_mixer']
        self.use_hgnn = config['use_hgnn']
        self.use_base_excitation = config['use_base_excitation']
        self.use_self_item_excitation = config['use_self_item_excitation']
        self.use_self_intent_excitation = config['use_self_intent_excitation']

        self.device = config['device']
        self.loss_type = config['loss_type']
        self.n_users = dataset.num(self.USER_ID)
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.time_field = config['TIME_FIELD']

        self.time_scalar = 60 * 60 * 24 * self.day_factor

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size)
        self.hyper_item_embedding = None
        self.item2intent = None
        self.intent_embedding = None

        # hawkes process
        self.global_alpha = nn.Parameter(torch.tensor(0.))
        self.dist_params = nn.Linear(2 * self.hidden_size, 5)

        self.alphas = nn.Embedding(self.n_items, 1)
        self.pis = nn.Embedding(self.n_items, 1)
        self.betas = nn.Embedding(self.n_items, 1)
        self.sigmas = nn.Embedding(self.n_items, 1)
        self.mus = nn.Embedding(self.n_items, 1)

        # hgnn
        if self.use_hgnn and self.hgnn_layers > 0:
            self.biparpite_graph = GeneralGraphDataset(config).build()[0].get_bipartite_time_inter_mat()
            self.hyper_edge_index, self.hyper_edge_weight = self.construct_global_hyper_graph(self.biparpite_graph)
            self.hgnn = HGNN(hidden_size=self.hidden_size, n_layers=self.hgnn_layers)

        # local encoding
        self.local_encoder = AttentionMixer(hidden_size=self.hidden_size, levels=self.n_levels, n_heads=self.n_heads,
                                            dropout=self.attn_dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.last_linear = nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "BCE":
            self.loss_fct = BCELoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'BCE']!")
        

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def construct_global_hyper_graph(self, biparpite_graph):
        bi_edge_index, bi_edge_weight, bi_edge_time = biparpite_graph

        self.logger.info('Constructing global hyper graph.')
        # get all subseqs
        offset = 0
        current_u_length = 0
        current_uid = bi_edge_index[0, 0]
        all_subseqs = []
        for uid in bi_edge_index[0, :]:
            if uid != current_uid:
                items = bi_edge_index[1, offset:offset + current_u_length]
                user_edge_times = bi_edge_time[offset:offset + current_u_length]
                edge_time_shifted = torch.roll(user_edge_times, 1)
                edge_time_shifted[0] = user_edge_times[0]
                time_delta = user_edge_times - edge_time_shifted
                sub_seqs = (time_delta > self.sub_time_delta).nonzero(as_tuple=True)[0]
                sub_seqs = torch.tensor_split(items, sub_seqs)
                all_subseqs += sub_seqs

                current_uid = uid
                offset += current_u_length
                current_u_length = 1
            else:
                current_u_length += 1

        # create co-occurence dict
        co_occurrence_dict = {}
        for seq in all_subseqs:
            for item in seq:
                item = int(item)
                if item not in co_occurrence_dict:
                    co_occurrence_dict[item] = {}
                co_occuring_items = seq[torch.where(seq != item)]
                for c_item in co_occuring_items:
                    c_item = int(c_item)
                    if c_item not in co_occurrence_dict[item]:
                        co_occurrence_dict[item][c_item] = 1
                    else:
                        co_occurrence_dict[item][c_item] += 1

        # find and filter intents
        # possible multiple intents per session -> subintents
        intents = []
        for seq in all_subseqs:
            sub_intents = []
            for item in seq:
                item = int(item)
                intent = set([item])
                co_occuring_items = seq[torch.where(seq != item)]
                for c_item in co_occuring_items:
                    c_item = int(c_item)
                    # item co-occur in at least one other session
                    if co_occurrence_dict[item][c_item] >= 2:
                        intent.add(c_item)
                # intent consists at least of 2 items
                if intent not in sub_intents and len(intent) >= 2:
                    sub_intents.append(intent)

            # add only superset intents
            superset_intents = sub_intents.copy()
            for sub_intent in sub_intents:
                for comp_subset in sub_intents:
                    if sub_intent < comp_subset:
                        superset_intents.remove(sub_intent)
                        break
            intents += superset_intents

        intent_counter = Counter(frozenset(s) for s in intents)

        nodes = []
        edges = []
        edge_weights = []
        for edge_index, items in enumerate(intent_counter):
            n_items = len(items)
            edge_weight = intent_counter[items]
            # filter out noisy, unreliable intents
            # if edge_weight > 1:
            nodes.append(torch.LongTensor(list(items)))
            edges.append(torch.LongTensor([edge_index] * n_items))
            edge_weights.append(torch.LongTensor([edge_weight] * n_items))

        # add self loops
        max_edge = len(intent_counter)
        for item in range(self.n_items):
            nodes.append(torch.LongTensor([item]))
            edges.append(torch.LongTensor([max_edge]))
            edge_weights.append(torch.LongTensor([1]))
            max_edge += 1

        edge_index = torch.stack((torch.cat(nodes), torch.cat(edges)))
        edge_weights = torch.cat(edge_weights)  # row normalized in conv

        return edge_index.to(self.device), edge_weights.to(self.device)

    def e_step(self):
        if self.use_hgnn and self.hgnn_layers > 0:
            self.hyper_item_embedding = self.hgnn(self.item_embedding.weight.detach(), self.hyper_edge_index,
                                                  self.hyper_edge_weight)

        if self.n_clusters >= self.n_items:
            self.logger.info("No clustering, since n_cluster >= n_items")
            self.item2intent = torch.arange(0, self.n_items).to(self.device)
            if self.hyper_item_embedding is not None:
                self.intent_embedding = self.hyper_item_embedding.detach().cpu().numpy()
            else:
                self.intent_embedding = self.item_embedding.weight.detach().cpu().numpy()
        else:
            if self.hyper_item_embedding is not None:
                item_embeddings = self.hyper_item_embedding.detach().cpu().numpy()
            else:
                item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
            self.item2intent, self.intent_embedding = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.hidden_size, k=self.n_clusters, gpu=True)
        kmeans.cp.max_points_per_centroid = 4096
        kmeans.cp.min_points_per_centroid = 0
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return node2cluster, centroids

    def get_intervals(self, item, time, item_seq, time_seq):
        item_mask = (item_seq == item.unsqueeze(-1))

        delta_t = (time.reshape(-1, 1) - time_seq)
        delta_t = (delta_t / self.time_scalar) * item_mask

        # take only last intent (minimum time span)
        delta_t = torch.min(delta_t + (~item_mask * 10e9), dim=1).values * item_mask.sum(dim=1).bool()
        mask = (delta_t > 0)

        interval = torch.ones((item_seq.shape[0])).to(self.device) * -1
        interval[mask] = delta_t[mask]

        return interval.unsqueeze(-1)

    def intent_excitation(self, item, item_seq, time, time_seq, user_id, user_rep):
        target2intent = self.item2intent[item]
        target2intent_emb = self.intent_embedding[target2intent]
        seq2intent = self.item2intent[item_seq]

        item_mask = item_seq.gt(0)
        intent_mask = (seq2intent == target2intent.unsqueeze(-1)) * item_mask

        delta_t = (time.reshape(-1, 1) - time_seq)
        delta_mask = (delta_t > self.sub_time_delta) * intent_mask
        delta_t = (delta_t / self.time_scalar) * delta_mask

        # take only last intent (minimum time span)
        delta_t = torch.min(delta_t + (~delta_mask * 10e9), dim=1).values * delta_mask.sum(dim=1).bool()
        mask = ((delta_t > 0).float())

        user_emb = self.user_embedding(user_id) + user_rep
        dist_params = self.dist_params(torch.cat((target2intent_emb, user_emb), dim=1))

        mus = (dist_params[:, 0]).clamp(min=1e-10, max=10)
        sigmas = (dist_params[:, 1]).clamp(min=1e-10, max=10)

        alphas = (self.global_alpha + dist_params[:, 2]).clamp(min=0, max=1)

        betas = (dist_params[:, 3] + 1).clamp(min=1e-10, max=10)
        pis = dist_params[:, 4] + 0.5

        exp_dist = torch.distributions.exponential.Exponential(betas, validate_args=False)
        norm_dist = torch.distributions.normal.Normal(mus, sigmas)

        excitation = pis * exp_dist.log_prob(delta_t).exp() + (1 - pis) * norm_dist.log_prob(delta_t).exp()

        excitation = alphas * excitation * mask

        return excitation.unsqueeze(-1)

    def item_excitation(self, items, r_intervals):
        alphas = self.global_alpha + self.alphas(items)
        pis, mus = self.pis(items) + 0.5, self.mus(items) + 1
        betas = (self.betas(items) + 1).clamp(min=1e-10, max=10)
        sigmas = (self.sigmas(items) + 1).clamp(min=1e-10, max=10)
        mask = ((r_intervals >= 0).float())
        delta_t = (r_intervals * mask)
        norm_dist = torch.distributions.normal.Normal(mus, sigmas)
        exp_dist = torch.distributions.exponential.Exponential(betas, validate_args=False)
        decay = pis * exp_dist.log_prob(delta_t).exp() + (1 - pis) * norm_dist.log_prob(delta_t).exp()
        decay = norm_dist.log_prob(delta_t).exp()
        excitation = (alphas * decay * mask)

        return excitation

    def get_user_rep(self, user_id, item_seq, target_item):
        mask = item_seq.gt(0).unsqueeze(2)
        mask = torch.where(mask, 0.0, -10000.0)  # for stable softmax

        user_emb = self.user_embedding(user_id)

        # MOJITO, (FISSA), FISM, AdaCML
        item_seq_emb = self.item_embedding(item_seq)
        target_item_emb = self.item_embedding(target_item)

        user_emb = F.dropout(user_emb, self.emb_dropout_prob, training=self.training)
        item_seq_emb = F.dropout(item_seq_emb, self.emb_dropout_prob, training=self.training)
        target_item_emb = F.dropout(target_item_emb, self.emb_dropout_prob, training=self.training)

        # fism attentive vectors
        attn_scores = torch.matmul(item_seq_emb, target_item_emb.unsqueeze(2))
        attn_probs = torch.softmax(attn_scores + mask, dim=1).transpose(2, 1)  # [B, 1, seq_len]

        attn_user_rep = torch.matmul(attn_probs, item_seq_emb).squeeze(1)  # [B, dim]

        user_rep = attn_user_rep

        return user_rep

    def forward(self, item_seq, item_seq_len):
        # last subseq, local intent, short term
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.LayerNorm(item_seq_emb)
        item_seq_emb = F.dropout(item_seq_emb, self.emb_dropout_prob, training=self.training)

        last_item_emb = self.gather_indexes(item_seq_emb, item_seq_len - 1)

        seq_output = self.local_encoder(item_seq, item_seq_emb, item_seq_len)
        seq_output = self.last_linear(torch.cat([seq_output, last_item_emb], dim=1))

        return seq_output

    def calculate_loss(self, interaction):
        user_id = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        time = interaction[self.time_field]
        time_seq = interaction[self.time_field + "_list"]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        pos_score = torch.zeros_like(pos_items).float().to(self.device)
        neg_score = torch.zeros_like(neg_items).float().to(self.device)

        pos_user_rep = self.get_user_rep(user_id, item_seq, pos_items)
        neg_user_rep = self.get_user_rep(user_id, item_seq, neg_items)

        pos_items_emb = self.item_embedding(pos_items)
        neg_items_emb = self.item_embedding(neg_items)
        pos_items_emb = F.dropout(pos_items_emb, self.emb_dropout_prob, training=self.training)
        neg_items_emb = F.dropout(neg_items_emb, self.emb_dropout_prob, training=self.training)

        if self.use_atten_mixer:
            seq_output = self.forward(item_seq, item_seq_len)

            pos_score += torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score += torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]

        if self.use_base_excitation:
            pos_score += torch.sum(pos_user_rep * pos_items_emb, dim=-1)
            neg_score += torch.sum(neg_user_rep * neg_items_emb, dim=-1)

        if self.use_self_item_excitation:
            pos_intervals = self.get_intervals(pos_items, time, item_seq, time_seq)
            neg_intervals = self.get_intervals(neg_items, time, item_seq, time_seq)

            p_item_excitation = self.item_excitation(pos_items, pos_intervals)
            n_item_excitation = self.item_excitation(neg_items, neg_intervals)

            pos_score += p_item_excitation.squeeze()
            neg_score += n_item_excitation.squeeze()

        if self.use_self_intent_excitation and self.item2intent is not None:
            p_excitation = self.intent_excitation(pos_items, item_seq, time, time_seq, user_id, pos_user_rep)
            n_excitation = self.intent_excitation(neg_items, item_seq, time, time_seq, user_id, neg_user_rep)

            pos_score += p_excitation.squeeze()
            neg_score += n_excitation.squeeze()

        loss = self.loss_fct(pos_score, neg_score)
        return loss

    def predict(self, interaction):
        user_id = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time = interaction[self.time_field]
        time_seq = interaction[self.time_field + "_list"]
        test_item = interaction[self.ITEM_ID]

        scores = torch.zeros_like(test_item).float().to(self.device)
        test_item_emb = self.item_embedding(test_item)
        user_rep = self.get_user_rep(user_id, item_seq, test_item)

        if self.use_atten_mixer:
            seq_output = self.forward(item_seq, item_seq_len)
            scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        if self.use_base_excitation:
            scores += torch.sum(user_rep * test_item_emb, dim=-1)

        if self.use_self_item_excitation:
            intervals = self.get_intervals(test_item, time, item_seq, time_seq)
            item_excitation = self.item_excitation(test_item, intervals)
            scores += item_excitation.squeeze()

        if self.use_self_intent_excitation and self.item2intent is not None:
            intent_excitation = self.intent_excitation(test_item, item_seq, time, time_seq, user_id, user_rep)
            scores += intent_excitation.squeeze()

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        time = interaction[self.time_field]
        time_seq = interaction[self.time_field + "_list"]
        item_ids = torch.arange(0, self.n_items, dtype=torch.long).to(self.device)

        scores = torch.zeros((user_id.size(0), item_ids.size(0))).float().to(self.device)

        if self.use_atten_mixer:
            seq_output = self.forward(item_seq, item_seq_len)
            test_items_emb = self.item_embedding.weight
            scores += torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]

        temporal_scores = []
        if (self.use_base_excitation or
                self.use_self_item_excitation or
                (self.use_self_intent_excitation and self.item2intent is not None)):
            for iid in item_ids.unsqueeze(-1):
                iid_batch = iid.repeat(user_id.size(0))
                test_item_emb = self.item_embedding(iid_batch)
                user_rep = self.get_user_rep(user_id, item_seq, iid_batch)
                excitation = torch.zeros_like(iid_batch).float()
                if self.use_base_excitation:
                    excitation += torch.sum(user_rep * test_item_emb, dim=-1)

                if self.use_self_item_excitation:
                    intervals = self.get_intervals(iid_batch, time, item_seq, time_seq)
                    item_excitation = self.item_excitation(iid_batch, intervals)
                    excitation += item_excitation.squeeze()

                if self.use_self_intent_excitation and self.item2intent is not None:
                    intent_excitation = self.intent_excitation(iid_batch, item_seq, time, time_seq, user_id, user_rep)
                    excitation += intent_excitation.squeeze()
                temporal_scores.append(excitation.squeeze())
            temporal_scores = torch.stack(temporal_scores, dim=1)

        if len(temporal_scores):
            scores = scores + temporal_scores

        return scores