r"""
HyperHawkes
################################################

Reference:
    Peintner et al. "Hypergraph-based Temporal Modelling of Repeated Intent for Sequential Recommendation." in WWW 2025.

"""
import numpy as np
import torch

from torch import nn
import torch.nn.functional as F

from recbole_custom.model.utils import construct_global_hyper_graph, construct_global_graph
from recbole_custom.utils import InputType
from recbole_custom.data.utils import get_bipartite_time_inter_mat
from recbole_custom.model.loss import BPRLoss, BCELoss
from recbole_custom.model.abstract_recommender import SequentialRecommender
from recbole_custom.model.init import xavier_uniform_initialization
from recbole_custom.model.layers import HGNN, AttentionMixer, GCN

from torch_kmeans import SoftKMeans


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
        self.temp_cluster = config['temp_cluster']
        self.min_support = config['min_support']

        self.use_hgnn = config['use_hgnn']
        self.use_gcn = config['use_gcn']
        self.use_shortterm = config['use_shortterm']
        self.use_base_excitation = config['use_base_excitation']
        self.use_self_intent_excitation = config['use_self_intent_excitation']

        self.device = config['device']
        self.loss_type = config['loss_type']
        self.n_users = dataset.num(self.USER_ID)
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]

        self.time_scalar = 60 * 60 * 24 * self.day_factor 

        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        if self.use_base_excitation or self.use_self_intent_excitation:
            self.user_embedding = nn.Embedding(self.n_users, self.hidden_size)

        if self.use_self_intent_excitation:
            self.hyper_item_embedding = None
            self.item2intent = torch.zeros((self.n_items, ), dtype=torch.long).to(self.device)
            self.cluster_prob = torch.zeros((self.n_items, self.n_clusters)).to(self.device)

        if self.use_self_intent_excitation:
            self.global_alpha = nn.Parameter(torch.tensor(0.))
            self.intent_dist = nn.Sequential(
                    nn.Linear(self.n_clusters + self.hidden_size * 2, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, 5)
                )

        ## hgnn
        if (self.use_hgnn or self.use_gcn) and self.hgnn_layers > 0 and self.use_self_intent_excitation:
            # global hyper graph construction
            self.biparpite_graph = get_bipartite_time_inter_mat(dataset)
            self.logger.info('Constructing global hyper graph.')
            if self.use_gcn:
                self.edge_index, self.edge_weight = construct_global_graph(self.biparpite_graph,
                                                                                             self.sub_time_delta,
                                                                                             self.n_items,
                                                                                             self.logger)
                self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)
            else:
                self.hyper_edge_index, self.hyper_edge_weight = construct_global_hyper_graph(self.biparpite_graph,
                                                                                         self.sub_time_delta,
                                                                                         self.min_support,
                                                                                         self.n_items,
                                                                                         self.logger)
                self.hyper_edge_index, self.hyper_edge_weight = self.hyper_edge_index.to(self.device), self.hyper_edge_weight.to(self.device)
            # self.n_edges = self.hyper_edge_index[1].max() + 1
            if self.use_gcn:
                self.gcn = GCN(config, hidden_size=self.hidden_size, n_layers=self.hgnn_layers)
            else:
                self.hgnn = HGNN(config, hidden_size=self.hidden_size, n_layers=self.hgnn_layers)

        if self.use_shortterm:
            self.local_encoder = AttentionMixer(hidden_size=self.hidden_size, levels=self.n_levels, n_heads=self.n_heads,
                                            dropout=self.attn_dropout_prob)
            self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            self.last_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)


        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "BCE":
            self.loss_fct = BCELoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'BCE']!")

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        if self.use_self_intent_excitation:
            self.other_parameter_name = ["cluster_prob"]

    def e_step(self):
        with torch.no_grad():
            if self.use_hgnn and self.hgnn_layers > 0 and not self.use_gcn:
                item_embs = self.hgnn(self.item_embedding.weight.detach(), self.hyper_edge_index,
                                                      self.hyper_edge_weight)
            elif self.use_gcn:
                item_embs = self.gcn(self.item_embedding.weight.detach(), self.edge_index, self.edge_weight)
            else:
                item_embs = self.item_embedding.weight.detach()

            self.cluster_prob = self.run_soft_kmeans(item_embs)

    def run_soft_kmeans(self, x):
        model = SoftKMeans(n_clusters=self.n_clusters,
                           init_method='k-means++',
                           normalize='unit',
                           temp=1 / self.temp_cluster,  # package interprets temp the other way around
                           verbose=False)
        
        result = model(x.unsqueeze(0), k=self.n_clusters)
        avg_max = result.soft_assignment.squeeze().max(dim=1).values.mean()
        self.logger.info("SoftKMeans: %s" % avg_max)
        return result.soft_assignment.squeeze()

    def intent_excitation(self, item, item_seq, time, time_seq, user_id, user_rep):
        if self.cluster_prob.sum(): # check for warmup
            target_cluster_probs = self.cluster_prob[item]
            seq_cluster_probs = self.cluster_prob[item_seq]

            kl_div = F.kl_div(seq_cluster_probs.log(), target_cluster_probs.log().unsqueeze(1), reduction='none', log_target=True).sum(dim=2)
            intent_mask = (kl_div < 1e-12) * item_seq.gt(0)

            delta_t = (time.reshape(-1, 1) - time_seq)
            delta_mask = (delta_t > self.sub_time_delta) * intent_mask
            delta_t = (delta_t / self.time_scalar) * delta_mask

            # take only last intent (minimum time span)
            delta_t = torch.min(delta_t + (~delta_mask * 10e9), dim=1).values * delta_mask.sum(dim=1).bool()
            mask = ((delta_t > 0).float())

            item_emb = self.item_embedding(item)

            dist_params = self.intent_dist(torch.cat((target_cluster_probs, item_emb, user_rep), dim=1))

            mus = (dist_params[:, 0]).clamp(min=1e-10, max=10)
            sigmas = (dist_params[:, 1]).clamp(min=1e-10, max=10)
            alphas = (self.global_alpha + dist_params[:, 2])
            betas = (dist_params[:, 3] + 1).clamp(min=1e-10, max=10)
            pis = (dist_params[:, 4] + 0.5).clamp(min=1e-10, max=1)

            exp_dist = torch.distributions.exponential.Exponential(betas, validate_args=False)
            norm_dist = torch.distributions.normal.Normal(mus, sigmas)

            excitation = pis * exp_dist.log_prob(delta_t).exp() + (1 - pis) * norm_dist.log_prob(delta_t).exp()
            excitation = alphas * excitation * mask

            return excitation.unsqueeze(-1)
        else:
            return torch.zeros_like(item).float()
    

    def get_user_rep(self, user_id, item_seq, target_item):
        mask = item_seq.gt(0).unsqueeze(2)
        mask = torch.where(mask, 0.0, -10000.0)  # for stable softmax

        user_emb = self.user_embedding(user_id)

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

        if self.use_shortterm:
            seq_output = self.forward(item_seq, item_seq_len)

            pos_score = pos_score + torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = neg_score + torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]

        if self.use_base_excitation:
            pos_score = pos_score + torch.sum(pos_user_rep * pos_items_emb, dim=-1)
            neg_score = neg_score + torch.sum(neg_user_rep * neg_items_emb, dim=-1)

        if self.use_self_intent_excitation and self.item2intent is not None:
            p_excitation = self.intent_excitation(pos_items, item_seq, time, time_seq, user_id, pos_user_rep)
            n_excitation = self.intent_excitation(neg_items, item_seq, time, time_seq, user_id, neg_user_rep)

            pos_score = pos_score + p_excitation.squeeze()
            neg_score = neg_score + n_excitation.squeeze()

        loss = self.loss_fct(pos_score, neg_score)

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

        if self.use_shortterm:
            seq_output = self.forward(item_seq, item_seq_len)
            scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        if self.use_base_excitation:
            scores = scores + torch.sum(user_rep * test_item_emb, dim=-1)

        if self.use_self_intent_excitation and self.item2intent is not None:
            intent_excitation = self.intent_excitation(test_item, item_seq, time, time_seq, user_id, user_rep)
            scores = scores + intent_excitation.squeeze()

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        time = interaction[self.TIME]
        time_seq = interaction[self.TIME_SEQ]
        item_ids = torch.arange(0, self.n_items, dtype=torch.long).to(self.device)

        scores = torch.zeros((user_id.size(0), item_ids.size(0))).float().to(self.device)

        if self.use_shortterm:
            seq_output = self.forward(item_seq, item_seq_len)
            test_items_emb = self.item_embedding.weight[:-1]
            scores = scores + torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]

        temporal_scores = []
        if (self.use_base_excitation or
                (self.use_self_intent_excitation and self.item2intent is not None)):
            for iid in item_ids.unsqueeze(-1):
                iid_batch = iid.repeat(user_id.size(0))
                test_item_emb = self.item_embedding(iid_batch)
                user_rep = self.get_user_rep(user_id, item_seq, iid_batch)
                excitation = torch.zeros_like(iid_batch).float()
                if self.use_base_excitation:
                    excitation = excitation + torch.sum(user_rep * test_item_emb, dim=-1)

                if self.use_self_intent_excitation and self.item2intent is not None:
                    intent_excitation = self.intent_excitation(iid_batch, item_seq, time, time_seq, user_id, user_rep)
                    excitation = excitation + intent_excitation.squeeze()
                temporal_scores.append(excitation.squeeze())
            temporal_scores = torch.stack(temporal_scores, dim=1)

        if len(temporal_scores):
            scores = scores + temporal_scores

        return scores
