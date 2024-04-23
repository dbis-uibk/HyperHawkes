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
import random
import itertools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn.init import xavier_normal_, constant_
from torch_geometric.nn import DMoNPooling, GraphMultisetTransformer, TopKPooling, ASAPooling
from tqdm import tqdm

from recbole_custom.model.utils import construct_global_hyper_graph, DataAugmention
from recbole_custom.utils import InputType
from recbole_custom.data.utils import get_bipartite_time_inter_mat
from recbole_custom.model.loss import BPRLoss, EmbLoss, BCELoss, NCELoss, ClusterLoss
from recbole_custom.model.abstract_recommender import GeneralRecommender, SequentialRecommender
from recbole_custom.model.init import xavier_normal_initialization, xavier_uniform_initialization
from recbole_custom.model.layers import HGNN, AttentionMixer, MLPLayers

import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain, combinations
from torch_geometric.utils import degree, k_hop_subgraph, add_self_loops, to_torch_coo_tensor
import torch_geometric

import faiss


class HyperHawkes(SequentialRecommender):
    r"""
    HyperHawkes models user sequences via a global hyper graph and an item & intent-based hawkes process for recommendation.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HyperHawkes, self).__init__(config, dataset)

        self.TIME = config['TIME_FIELD']
        self.TIME_SEQ = config['TIME_FIELD'] + config["LIST_SUFFIX"]

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
        self.cluster_threshold = config['thresh_cluster']

        self.use_cl = config['use_cl']
        self.use_hgnn = config['use_hgnn']
        self.use_atten_mixer = config['use_atten_mixer']
        self.use_base_excitation = config['use_base_excitation']
        self.use_self_item_excitation = config['use_self_item_excitation']
        self.use_self_intent_excitation = config['use_self_intent_excitation']

        self.device = config['device']
        self.loss_type = config['loss_type']
        self.n_users = dataset.num(self.USER_ID)
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]

        self.time_scalar = 60 * 60 * 24 * self.day_factor  # max/mean? time difference in dataset

        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        if self.use_base_excitation or self.use_self_intent_excitation:
            self.user_embedding = nn.Embedding(self.n_users, self.hidden_size)

        if self.use_self_intent_excitation:
            self.hyper_item_embedding = None
            self.item2intent = torch.zeros((self.n_items, ), dtype=torch.long).to(self.device)
            self.cluster_prob = torch.zeros((self.n_items,)).to(self.device)

        if self.use_self_item_excitation or self.use_self_intent_excitation:
            self.global_alpha = nn.Parameter(torch.tensor(0.))

        if self.use_self_item_excitation:
            self.alphas = nn.Embedding(self.n_items, 1)
            self.pis = nn.Embedding(self.n_items, 1)
            self.betas = nn.Embedding(self.n_items, 1)
            self.sigmas = nn.Embedding(self.n_items, 1)
            self.mus = nn.Embedding(self.n_items, 1)

        if self.use_cl:
            self.intent_rep = torch.zeros(((self.n_items, self.n_clusters))).to(self.device)

            self.cluster_projector = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.n_clusters),
                nn.Softmax(dim=1)
            )

            self.cl_weight = config['cl_weight']
            self.temperature = config['temperature']
            self.cl_criterion = ClusterLoss(self.n_clusters, self.temperature, self.device)

            # self.user_alphas = nn.Embedding(self.n_users, 1)
            # self.intent_alphas = nn.Embedding(self.n_clusters, 1)
            # self.user_intent_alphas = nn.Linear(self.n_clusters + self.hidden_size, 1)

            self.intent_dist = nn.Linear(self.n_clusters + self.hidden_size, 5)
            # self.intent_dist = nn.Linear(self.n_clusters, 5)
            # self.intent_dist = nn.Linear(self.hidden_size, 5)

            # self.intent_pis = nn.Embedding(self.n_clusters, 1)
            # self.intent_betas = nn.Embedding(self.n_clusters, 1)
            # self.intent_sigmas = nn.Embedding(self.n_clusters, 1)
            # self.intent_mus = nn.Embedding(self.n_clusters, 1)

        ## hgnn
        if self.use_hgnn and self.hgnn_layers > 0:
            # global hyper graph construction
            self.biparpite_graph = get_bipartite_time_inter_mat(dataset)
            self.logger.info('Constructing global hyper graph.')
            self.hyper_edge_index, self.hyper_edge_weight = construct_global_hyper_graph(self.biparpite_graph,
                                                                                         self.sub_time_delta,
                                                                                         self.n_items)
            self.hyper_edge_index, self.hyper_edge_weight = self.hyper_edge_index.to(self.device), self.hyper_edge_weight.to(self.device)
            self.logger.info('Max. edge weight: %.4f' % self.hyper_edge_weight.max().item())
            self.logger.info('Sparsity of hyper graph: %.4f' % (1.0 - (self.hyper_edge_index.size(1) / (self.n_items ** 2))))
            # self.n_edges = self.hyper_edge_index[1].max() + 1
            self.hgnn = HGNN(config, hidden_size=self.hidden_size, n_layers=self.hgnn_layers)

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
        self.other_parameter_name = ["intent_rep", "item2intent", "cluster_prob"]

    def e_step(self):
        with torch.no_grad():
            if self.use_hgnn and self.hgnn_layers > 0:
                item_embs = self.hgnn(self.item_embedding.weight.detach(), self.hyper_edge_index,
                                                    self.hyper_edge_weight)
            else:
                item_embs = self.item_embedding.weight.detach()

            if self.use_cl:
                with torch.no_grad():
                    self.intent_rep, self.item2intent, self.cluster_prob = self.forward_cluster(item_embs)
                self.logger.info(f"Cluster confidence: {self.cluster_projector(item_embs).max(dim=1)[0].mean():.4f}")

    def contrastive_clustering(self, item_sub_seq, item_sub_seq_len, aug_item_sub_seq, aug_item_sub_seq_len):
        if self.use_hgnn and self.hgnn_layers > 0:
            item_embedding = self.hgnn(self.item_embedding.weight.detach(), self.hyper_edge_index,
                                         self.hyper_edge_weight)
        else:
            item_embedding = self.item_embedding.weight

        # get hyper session representations -> average of embs or last item hyper rep?
        item_seq_emb = item_embedding[item_sub_seq]
        mask = (item_sub_seq != 0).unsqueeze(-1)
        sess_rep = torch.sum(item_seq_emb * mask, -2) / torch.sum(mask, 1)
        #sess_rep = self.gather_indexes(item_seq_emb, item_sub_seq_len - 1)

        aug_item_seq_emb = item_embedding[aug_item_sub_seq]
        #aug_mask = ((aug_item_sub_seq != 0) & (aug_item_sub_seq != self.n_items)).unsqueeze(-1)
        aug_mask = (aug_item_sub_seq != 0).unsqueeze(-1)

        aug_sess_rep = torch.sum(aug_item_seq_emb * aug_mask, -2) / torch.sum(aug_mask, 1)
        #aug_sess_rep = self.gather_indexes(aug_item_seq_emb, aug_item_sub_seq_len - 1)

        # project to latent embeddings, disentangle intents
        h = self.cluster_projector(sess_rep)
        h_aug = self.cluster_projector(aug_sess_rep)

        cl_loss = self.cl_criterion(h, h_aug)

        return h, cl_loss

    def forward_cluster(self, item_embs):
        c_rep = self.cluster_projector(item_embs)
        c_prob, c = torch.max(c_rep, dim=1)
        return c_rep.detach(), c, c_prob

    def intent_excitation(self, item, item_seq, time, time_seq, user_id, user_rep):
        target2intent = self.item2intent[item]
        seq2prob = self.cluster_prob[item_seq]
        target2intent_emb = self.intent_rep[target2intent]
        seq2intent = self.item2intent[item_seq]

        # only use confident cluster assignments
        item_mask = item_seq.gt(0) * (seq2prob > self.cluster_threshold).gt(0)
        intent_mask = (seq2intent == target2intent.unsqueeze(-1)) * item_mask

        delta_t = (time.reshape(-1, 1) - time_seq)
        delta_mask = (delta_t > self.sub_time_delta) * intent_mask
        delta_t = (delta_t / self.time_scalar) * delta_mask

        # take only last intent (minimum time span)
        delta_t = torch.min(delta_t + (~delta_mask * 10e9), dim=1).values * delta_mask.sum(dim=1).bool()
        #delta_t = delta_t.unsqueeze(-1)
        mask = ((delta_t > 0).float())

        '''user_intent_alphas = self.user_intent_alphas(torch.cat((target2intent_emb, user_rep), dim=1)).clamp(min=1e-10, max=10)
        alphas = self.global_alpha + user_intent_alphas
        pis, mus = self.intent_pis(target2intent) + 0.5, self.intent_mus(target2intent) + 1
        betas = (self.intent_betas(target2intent) + 1).clamp(min=1e-10, max=10)
        sigmas = (self.intent_sigmas(target2intent) + 1).clamp(min=1e-10, max=10)'''

        dist_params = self.intent_dist(torch.cat((target2intent_emb, user_rep), dim=1))
        #dist_params = self.intent_dist(target2intent_emb)
        #dist_params = self.intent_dist(user_rep)

        mus = (dist_params[:, 0]).clamp(min=1e-10, max=10)
        sigmas = (dist_params[:, 1]).clamp(min=1e-10, max=10)
        alphas = (self.global_alpha + dist_params[:, 2])  # .clamp(min=0, max=1)
        betas = (dist_params[:, 3] + 1).clamp(min=1e-10, max=10)
        pis = dist_params[:, 4] + 0.5

        exp_dist = torch.distributions.exponential.Exponential(betas, validate_args=False)
        norm_dist = torch.distributions.normal.Normal(mus, sigmas)

        excitation = pis * exp_dist.log_prob(delta_t).exp() + (1 - pis) * norm_dist.log_prob(delta_t).exp()
        excitation = alphas * excitation * mask

        return excitation.unsqueeze(-1)

    def item_excitation(self, item, item_seq, time, time_seq):
        item_mask = (item_seq == item.unsqueeze(-1))

        delta_t = (time.reshape(-1, 1) - time_seq)
        delta_t = (delta_t / self.time_scalar) * item_mask

        # take only last item (minimum time span)
        delta_t = torch.min(delta_t + (~item_mask * 10e9), dim=1).values * item_mask.sum(dim=1).bool()
        delta_t = delta_t.unsqueeze(-1)
        mask = ((delta_t > 0).float())

        alphas = self.global_alpha + self.alphas(item)
        pis, mus = self.pis(item) + 0.5, self.mus(item) + 1
        betas = (self.betas(item) + 1).clamp(min=1e-10, max=10)
        sigmas = (self.sigmas(item) + 1).clamp(min=1e-10, max=10)

        norm_dist = torch.distributions.normal.Normal(mus, sigmas)
        exp_dist = torch.distributions.exponential.Exponential(betas, validate_args=False)
        decay = pis * exp_dist.log_prob(delta_t).exp() + (1 - pis) * norm_dist.log_prob(delta_t).exp()
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

        user_rep = attn_user_rep + user_emb

        return user_rep

    def forward(self, item_seq, item_seq_len):
        # last subseq, local intent, short term
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.LayerNorm(item_seq_emb)
        item_seq_emb = F.dropout(item_seq_emb, self.emb_dropout_prob, training=self.training)

        last_item = self.gather_indexes(item_seq.unsqueeze(-1), item_seq_len - 1).squeeze(1)
        last_item_emb = self.gather_indexes(item_seq_emb, item_seq_len - 1)

        seq_output = self.local_encoder(item_seq, item_seq_emb, item_seq_len)

        seq_output = self.last_linear(torch.cat([seq_output, last_item_emb], dim=1))

        return seq_output

    def calculate_loss(self, interaction):
        user_id = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        time = interaction[self.TIME]
        time_seq = interaction[self.TIME_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        pos_score = torch.zeros_like(pos_items).float().to(self.device)
        neg_score = torch.zeros_like(neg_items).float().to(self.device)

        if self.use_base_excitation or self.use_self_intent_excitation:
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
            p_item_excitation = self.item_excitation(pos_items, item_seq, time, time_seq)
            n_item_excitation = self.item_excitation(neg_items, item_seq, time, time_seq)

            pos_score += p_item_excitation.squeeze()
            neg_score += n_item_excitation.squeeze()

        if self.use_self_intent_excitation and self.item2intent is not None:
            p_excitation = self.intent_excitation(pos_items, item_seq, time, time_seq, user_id, pos_user_rep)
            n_excitation = self.intent_excitation(neg_items, item_seq, time, time_seq, user_id, neg_user_rep)

            pos_score += p_excitation.squeeze()
            neg_score += n_excitation.squeeze()

        loss = self.loss_fct(pos_score, neg_score)

        if self.use_cl:
            item_sub_seq = interaction["sub_" + self.ITEM_SEQ]
            item_sub_seq_len = interaction["sub_" + self.ITEM_SEQ_LEN]
            aug_item_sub_seq = interaction["mask_sub_" + self.ITEM_SEQ]
            aug_item_sub_seq_len = interaction["mask_sub_" + self.ITEM_SEQ_LEN]
            cluster_rep, cl_loss = self.contrastive_clustering(item_sub_seq, item_sub_seq_len, aug_item_sub_seq, aug_item_sub_seq_len)

            return loss + (self.cl_weight * cl_loss)

        return loss

    def predict(self, interaction):
        user_id = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time = interaction[self.TIME]
        time_seq = interaction[self.TIME_SEQ]
        test_item = interaction[self.ITEM_ID]

        scores = torch.zeros_like(test_item).float().to(self.device)
        test_item_emb = self.item_embedding(test_item)
        if self.use_base_excitation or self.use_self_intent_excitation:
            user_rep = self.get_user_rep(user_id, item_seq, test_item)

        if self.use_atten_mixer:
            seq_output = self.forward(item_seq, item_seq_len)
            scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        if self.use_base_excitation:
            scores += torch.sum(user_rep * test_item_emb, dim=-1)

        if self.use_self_item_excitation:
            item_excitation = self.item_excitation(test_item, item_seq, time, time_seq)
            scores += item_excitation.squeeze()

        if self.use_self_intent_excitation and self.item2intent is not None:
            intent_excitation = self.intent_excitation(test_item, item_seq, time, time_seq, user_id, user_rep)
            scores += intent_excitation.squeeze()

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        time = interaction[self.TIME]
        time_seq = interaction[self.TIME_SEQ]
        item_ids = torch.arange(0, self.n_items, dtype=torch.long).to(self.device)

        scores = torch.zeros((user_id.size(0), item_ids.size(0))).float().to(self.device)

        if self.use_atten_mixer:
            seq_output = self.forward(item_seq, item_seq_len)
            test_items_emb = self.item_embedding.weight[:-1]
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
                    item_excitation = self.item_excitation(iid_batch, item_seq, time, time_seq)
                    excitation += item_excitation.squeeze()

                if self.use_self_intent_excitation and self.item2intent is not None:
                    intent_excitation = self.intent_excitation(iid_batch, item_seq, time, time_seq, user_id, user_rep)
                    excitation += intent_excitation.squeeze()
                temporal_scores.append(excitation.squeeze())
            temporal_scores = torch.stack(temporal_scores, dim=1)

        if len(temporal_scores):
            scores = scores + temporal_scores

        return scores

    def export_intents(self):
        item_names = None
        if self.dataset.item_feat:
            item_names = self.dataset.field2id_token['item_name'][self.dataset.item_feat.item_name]
            vendor_ids = self.dataset.field2id_token['vendor_id'][self.dataset.item_feat.vendor_id]

        hyper_embs = self.hgnn(self.item_embedding.weight,
                               self.hyper_edge_index,
                               self.hyper_edge_weight).detach().cpu().numpy()

        embs = self.item_embedding.weight.detach().cpu().numpy()

        clustering = self.item2intent.cpu().numpy()
        #intervals = (self.intent_dist(torch.cat((self.intent_rep, self.user_embedding.weight[:self.intent_rep.size(0)]), dim=1))[:, 0] * self.day_factor).detach().cpu().numpy()
        intervals = 0

        return item_names, vendor_ids, hyper_embs, embs, clustering, intervals
