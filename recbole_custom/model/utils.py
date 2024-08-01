import math
import random
import numpy as np
import pandas as pd
from pyrsistent import v
import torch

from collections import defaultdict, Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax


class DataAugmention:
    def __init__(self, n_items, sub_time_delta, eta=0.6, gamma=0.3, beta=0.6):
        self.n_items = n_items
        self.sub_time_delta = sub_time_delta
        self.eta = eta
        self.gamma = gamma
        self.beta = beta

    def augment(self, item_seq, item_seq_len, time_seq):
        time_seq_shifted = torch.roll(time_seq, 1)
        time_seq_shifted[:, 0] = time_seq[:, 0]
        time_delta = torch.abs(time_seq - time_seq_shifted)
        sub_seq_mask = torch.cumsum((time_delta > self.sub_time_delta), dim=1)

        all_subseqs = []
        for seq, length, mask in zip(item_seq, item_seq_len, sub_seq_mask):
            subseqs = mask[:length].unique()
            for session in subseqs:
                subseq_index = (mask == session).nonzero(as_tuple=True)[0]
                subseq = seq[subseq_index]
                if len(subseq) > 1: # TODO needed?
                    all_subseqs.append(subseq)

        all_subseqs = torch.nn.utils.rnn.pad_sequence(all_subseqs, batch_first=True, padding_value=0.0)
        all_subseqs_lenghts = torch.count_nonzero(all_subseqs, dim=1)

        aug_seqs = []
        aug_lens = []
        for seq, length in zip(all_subseqs, all_subseqs_lenghts):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length

            if switch[0] == 0:
                aug_seq, aug_len = self.item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = self.item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = self.item_reorder(seq, length)

            aug_seqs.append(aug_seq)
            aug_lens.append(aug_len)

        return all_subseqs, all_subseqs_lenghts, torch.stack(aug_seqs), torch.stack(aug_lens)

    def item_crop(self, item_seq, item_seq_len):
        num_left = max(1, math.floor(item_seq_len * self.eta))
        crop_begin = random.randint(0, item_seq_len - num_left)
        croped_item_seq = np.zeros(item_seq.shape[0])
        if crop_begin + num_left < item_seq.shape[0]:
            croped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
        else:
            croped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:]
        return torch.tensor(croped_item_seq, dtype=torch.long, device=item_seq.device), \
            torch.tensor(num_left, dtype=torch.long, device=item_seq.device)

    def item_mask(self, item_seq, item_seq_len):
        num_mask = max(1, math.floor(item_seq_len * self.gamma))
        mask_index = random.sample(range(item_seq_len), k=num_mask)
        masked_item_seq = item_seq.cpu().detach().numpy().copy()
        masked_item_seq[mask_index] = self.n_items  # token 0 has been used for semantic masking
        return torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len

    def item_reorder(self, item_seq, item_seq_len):
        num_reorder = max(1, math.floor(item_seq_len * self.beta))
        reorder_begin = random.randint(0, item_seq_len - num_reorder)
        reordered_item_seq = item_seq.cpu().detach().numpy().copy()
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
        return torch.tensor(reordered_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len


def get_sub_sequences(biparpite_graph, sub_time_delta):
    bi_edge_index, bi_edge_weight, bi_edge_time = biparpite_graph

    # get all subseqs/sessions
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
            sub_seqs = (time_delta > sub_time_delta).nonzero(as_tuple=True)[0]
            sub_seqs = torch.tensor_split(items, sub_seqs)
            all_subseqs += [s.tolist() for s in sub_seqs]

            current_uid = uid
            offset += current_u_length
            current_u_length = 1
        else:
            current_u_length += 1

    return all_subseqs
def construct_global_graph(biparpite_graph, sub_time_delta, n_nodes, logger):
    # get all subseqs/sessions
    all_subseqs = get_sub_sequences(biparpite_graph, sub_time_delta)

    graph = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, seq in enumerate(all_subseqs):
        if len(seq) > 1:
            for j in range(len(seq) - 1):
                graph[seq[j], seq[j + 1]] += 1

    edge_index = torch.tensor(np.nonzero(graph), dtype=torch.long)
    edge_weights = torch.tensor(graph[edge_index[0], edge_index[1]], dtype=torch.float)

    return edge_index, edge_weights

def construct_global_hyper_graph(biparpite_graph, sub_time_delta, min_support, n_nodes, logger):
    # get all subseqs/sessions
    all_subseqs = get_sub_sequences(biparpite_graph, sub_time_delta)

    # intent rule mining
    te = TransactionEncoder()
    te_ary = te.fit(all_subseqs).transform(all_subseqs)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    logger.info(f"len(all_subseqs): {len(all_subseqs)}")
    
    frequent_itemsets = fpmax(df_encoded, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets = frequent_itemsets[(frequent_itemsets['length'] >= 2)].reset_index(drop=True)
    logger.info(f"# mined sets: {len(frequent_itemsets)}")

    # build hyper graph
    nodes = []
    edges = []
    edge_weights = []
    max_edge_weight = frequent_itemsets["support"].max()
    min_edge_weight = frequent_itemsets["support"].min()
    for edge_id, row in frequent_itemsets.iterrows():
        items = row["itemsets"]
        n_items = len(items)
        edge_support = row["support"]
        edge_weight = 1 + 99 * (edge_support - min_edge_weight) / (max_edge_weight - min_edge_weight) # scale edge weight to [1, 100]
        nodes.append(torch.LongTensor(list(items)))
        edges.append(torch.LongTensor([edge_id] * n_items))
        edge_weights.append(torch.FloatTensor([edge_weight] * n_items))

    if len(nodes) == 0:
        logger.info("No frequent itemsets found")
        return torch.tensor([[0, 0], [0, 0]]), torch.tensor([1])

    edge_index = torch.stack((torch.cat(nodes), torch.cat(edges)))
    edge_weights = torch.cat(edge_weights)  # row normalized in conv

    # add self loops
    # already done in conv manually
    '''num_edge = edge_index[1].max().item() + 1
    loop_index = torch.range(0, n_nodes, dtype=torch.long)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    loop_index[1] = num_edge + torch.range(0, n_nodes, dtype=torch.long)
    loop_index = loop_index.repeat_interleave(2, dim=1)
    loop_weight = torch.ones(n_nodes, dtype=torch.float)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    edge_weights = torch.cat([edge_weights, loop_weight], dim=0)'''

    logger.info(f"Number of edges: {torch.unique(edge_index[1]).size(0)}")
    logger.info('Max. edge weight: %.4f' % edge_weights.max().item())
    logger.info('Sparsity of hyper graph: %.6f' % (1.0 - (edge_index.size(1) / (n_nodes ** 2))))

    return edge_index, edge_weights
