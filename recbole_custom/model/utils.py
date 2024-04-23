import math
import random
import numpy as np
import torch

from collections import defaultdict, Counter


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


def construct_global_hyper_graph(biparpite_graph, sub_time_delta, n_items):
    bi_edge_index, bi_edge_weight, bi_edge_time = biparpite_graph

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
            sub_seqs = (time_delta > sub_time_delta).nonzero(as_tuple=True)[0]
            sub_seqs = torch.tensor_split(items, sub_seqs)
            all_subseqs += [s.tolist() for s in sub_seqs]

            current_uid = uid
            offset += current_u_length
            current_u_length = 1
        else:
            current_u_length += 1

    #intents = [set(x) for x in all_subseqs]

    # create co-occurence dict
    co_occurrence_dict = defaultdict(Counter)
    for seq in all_subseqs:
        for item in set(seq):
            co_occuring_items = [c_item for c_item in seq if c_item != item]
            co_occurrence_dict[item].update(co_occuring_items)

    # find and filter intents
    # possible multiple intents per session -> subintents
    intents = []
    for seq in all_subseqs:
        sub_intents = []
        unique_items = set(seq)
        for item in unique_items:
            intent = {item}
            co_occuring_items = unique_items - intent
            intent.update([c_item for c_item in co_occuring_items if co_occurrence_dict[item][c_item] >= 2])
            if len(intent) >= 2:
                sub_intents.append(intent)

        # add only superset intents
        sub_intents.sort(key=len, reverse=True)
        superset_intents = set(frozenset(i) for i in sub_intents)
        for sub_intent in sub_intents:
            for comp_subset in sub_intents:
                if sub_intent < comp_subset:
                    superset_intents.discard(frozenset(sub_intent))
                    break
        intents += list(superset_intents)
        #intents += sub_intents

    intent_counter = Counter(frozenset(s) for s in intents)

    nodes = []
    edges = []
    edge_weights = []
    for edge_id, items in enumerate(intent_counter):
        n_items = len(items)
        edge_weight = intent_counter[items]
        # filter out noisy, unreliable intents
        # if edge_weight > 1:
        nodes.append(torch.LongTensor(list(items)))
        edges.append(torch.LongTensor([edge_id] * n_items))
        edge_weights.append(torch.LongTensor([edge_weight] * n_items))

    edge_index = torch.stack((torch.cat(nodes), torch.cat(edges)))
    edge_weights = torch.cat(edge_weights)  # row normalized in conv

    # add self loops
    num_nodes = edge_index[0].max().item() + 1
    num_edge = edge_index[1].max().item() + 1
    loop_index = torch.arange(0, num_nodes, dtype=torch.long)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    loop_index[1] += num_edge
    loop_index = loop_index.repeat_interleave(2, dim=1)
    loop_weight = torch.ones(num_nodes, dtype=torch.float)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    edge_weights = torch.cat([edge_weights, loop_weight], dim=0)

    #print(f"Number of edges: {edge_index[1].max().item() + 1}")
    #print(f"Number of nodes: {edge_index[0].max().item() + 1}")

    return edge_index, edge_weights