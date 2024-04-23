# -*- coding: utf-8 -*-
# @Time   : 2022/7/19
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com

import math
import numpy as np
import random
import torch
from copy import deepcopy
from recbole_custom.data.interaction import Interaction, cat_interactions


def construct_transform(config):
    """
    Transformation for batch data.
    """
    if config["transform"] is None:
        return Equal(config)
    else:
        str2transform = {
            "mask_itemseq": MaskItemSequence,
            "multi_aug_itemseq": MultiAugItemSequence,
            "session_mask_itemseq": SessionMaskItemSequence,
            "inverse_itemseq": InverseItemSequence,
            "crop_itemseq": CropItemSequence,
            "reorder_itemseq": ReorderItemSequence,
            "user_defined": UserDefinedTransform,
        }
        if config["transform"] not in str2transform:
            raise NotImplementedError(
                f"There is no transform named '{config['transform']}'"
            )

        return str2transform[config["transform"]](config)


class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        return interaction


class MultiAugItemSequence:
    """
    Random augmentations for item sequence.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.AUG_ITEM_SEQ = "Aug_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.AUG_ITEM_SEQ_LEN = "Aug_" + self.ITEM_SEQ_LEN

        self.mask_ratio = config["mask_ratio"]
        self.crop_eta = config["eta"]
        self.reorder_beta = config["beta"]

        self.item_mask = MaskItemSequence(config=config)
        self.item_crop = CropItemSequence(config=config)
        self.item_reorder = ReorderItemSequence(config=config)

        self.AUG_ITEM_SEQ_1 = self.AUG_ITEM_SEQ + '_1'
        self.AUG_ITEM_SEQ_LEN_1 = self.AUG_ITEM_SEQ_LEN + '_1'
        self.AUG_ITEM_SEQ_2 = self.AUG_ITEM_SEQ + '_2'
        self.AUG_ITEM_SEQ_LEN_2 = self.AUG_ITEM_SEQ_LEN + '_2'

        config["AUG_ITEM_SEQ_1"] = self.AUG_ITEM_SEQ_1
        config["AUG_ITEM_SEQ_LEN_1"] = self.AUG_ITEM_SEQ_LEN_1
        config["AUG_ITEM_SEQ_2"] = self.AUG_ITEM_SEQ_2
        config["AUG_ITEM_SEQ_LEN_2"] = self.AUG_ITEM_SEQ_LEN_2

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        mask_item_seq = self.item_mask(dataset, interaction)[self.item_mask.MASK_ITEM_SEQ]
        crop_item_seq = self.item_crop(dataset, interaction)[self.item_crop.CROP_ITEM_SEQ]
        reorder_item_seq = self.item_reorder(dataset, interaction)[self.item_reorder.REORDER_ITEM_SEQ]

        aug_seq1, aug_seq2 = [], []
        for i, (seq, length) in enumerate(zip(item_seq, item_seq_len)):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq

            if switch[0] == 0:
                aug_seq = crop_item_seq[i]
            elif switch[0] == 1:
                aug_seq = mask_item_seq[i]
            elif switch[0] == 2:
                aug_seq = reorder_item_seq[i]

            aug_seq1.append(aug_seq)

            if switch[1] == 0:
                aug_seq = crop_item_seq[i]
            elif switch[1] == 1:
                aug_seq = mask_item_seq[i]
            elif switch[1] == 2:
                aug_seq = reorder_item_seq[i]

            aug_seq2.append(aug_seq)

        aug_seq1 = torch.stack(aug_seq1)
        aug_seq2 = torch.stack(aug_seq2)

        aug_seq_len1 = aug_seq1.bool().sum(dim=1)
        aug_seq_len2 = aug_seq2.bool().sum(dim=1)

        new_dict = {
            self.AUG_ITEM_SEQ_1: aug_seq1,
            self.AUG_ITEM_SEQ_LEN_1: aug_seq_len1,
            self.AUG_ITEM_SEQ_2: aug_seq2,
            self.AUG_ITEM_SEQ_LEN_2: aug_seq_len2,
        }
        interaction.update(Interaction(new_dict))
        return interaction


class MaskItemSequence:
    """
    Mask item sequence for training.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.MASK_ITEM_SEQ = "Mask_" + self.ITEM_SEQ
        self.POS_ITEMS = "Pos_" + config["ITEM_ID_FIELD"]
        self.NEG_ITEMS = "Neg_" + config["ITEM_ID_FIELD"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.mask_ratio = config["mask_ratio"]
        self.ft_ratio = 0 if not hasattr(config, "ft_ratio") else config["ft_ratio"]
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        self.MASK_INDEX = "MASK_INDEX"
        config["MASK_INDEX"] = "MASK_INDEX"
        config["MASK_ITEM_SEQ"] = self.MASK_ITEM_SEQ
        config["POS_ITEMS"] = self.POS_ITEMS
        config["NEG_ITEMS"] = self.NEG_ITEMS
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.config = config

    def _neg_sample(self, item_set, n_items):
        item = random.randint(1, n_items - 1)
        while item in item_set:
            item = random.randint(1, n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def _append_mask_last(self, interaction, n_items, device):
        batch_size = interaction[self.ITEM_SEQ].size(0)
        pos_items, neg_items, masked_index, masked_item_sequence = [], [], [], []
        seq_instance = interaction[self.ITEM_SEQ].cpu().numpy().tolist()
        item_seq_len = interaction[self.ITEM_SEQ_LEN].cpu().numpy().tolist()
        for instance, lens in zip(seq_instance, item_seq_len):
            mask_seq = instance.copy()
            ext = instance[lens - 1]
            mask_seq[lens - 1] = n_items
            masked_item_sequence.append(mask_seq)
            pos_items.append(self._padding_sequence([ext], self.mask_item_length))
            neg_items.append(
                self._padding_sequence(
                    [self._neg_sample(instance, n_items)], self.mask_item_length
                )
            )
            masked_index.append(
                self._padding_sequence([lens - 1], self.mask_item_length)
            )
        # [B Len]
        masked_item_sequence = torch.tensor(
            masked_item_sequence, dtype=torch.long, device=device
        ).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        new_dict = {
            self.MASK_ITEM_SEQ: masked_item_sequence,
            self.POS_ITEMS: pos_items,
            self.NEG_ITEMS: neg_items,
            self.MASK_INDEX: masked_index,
        }
        ft_interaction = deepcopy(interaction)
        ft_interaction.update(Interaction(new_dict))
        return ft_interaction

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        device = item_seq.device
        batch_size = item_seq.size(0)
        n_items = dataset.num(self.ITEM_ID)
        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []

        if random.random() < self.ft_ratio:
            interaction = self._append_mask_last(interaction, n_items, device)
        else:
            for instance in sequence_instances:
                # WE MUST USE 'copy()' HERE!
                masked_sequence = instance.copy()
                pos_item = []
                neg_item = []
                index_ids = []
                for index_id, item in enumerate(instance):
                    # padding is 0, the sequence is end
                    if item == 0:
                        break
                    prob = random.random()
                    if prob < self.mask_ratio:
                        pos_item.append(item)
                        neg_item.append(self._neg_sample(instance, n_items))
                        masked_sequence[index_id] = n_items
                        index_ids.append(index_id)

                masked_item_sequence.append(masked_sequence)
                pos_items.append(
                    self._padding_sequence(pos_item, self.mask_item_length)
                )
                neg_items.append(
                    self._padding_sequence(neg_item, self.mask_item_length)
                )
                masked_index.append(
                    self._padding_sequence(index_ids, self.mask_item_length)
                )

            # [B Len]
            masked_item_sequence = torch.tensor(
                masked_item_sequence, dtype=torch.long, device=device
            ).view(batch_size, -1)
            # [B mask_len]
            pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(
                batch_size, -1
            )
            # [B mask_len]
            neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(
                batch_size, -1
            )
            # [B mask_len]
            masked_index = torch.tensor(
                masked_index, dtype=torch.long, device=device
            ).view(batch_size, -1)
            new_dict = {
                self.MASK_ITEM_SEQ: masked_item_sequence,
                self.POS_ITEMS: pos_items,
                self.NEG_ITEMS: neg_items,
                self.MASK_INDEX: masked_index,
            }
            interaction.update(Interaction(new_dict))
        return interaction


class SessionMaskItemSequence:
    """
    Split into sub-sequences (sessions) and mask item sequence for training.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.TIME_SEQ = config['TIME_FIELD'] + config["LIST_SUFFIX"]

        self.ITEM_SUB_SEQ = "sub_" + self.ITEM_SEQ
        self.ITEM_SUB_SEQ_LEN = "sub_" + self.ITEM_SEQ_LEN
        self.MASK_ITEM_SUB_SEQ = "mask_" + self.ITEM_SUB_SEQ
        self.MASK_ITEM_SUB_SEQ_LEN = "mask_" + self.ITEM_SUB_SEQ_LEN

        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.mask_ratio = config["mask_ratio"]
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        config["ITEM_SUB_SEQ"] = self.ITEM_SUB_SEQ
        config["ITEM_SUB_SEQ_LEN"] = self.ITEM_SUB_SEQ_LEN
        config["MASK_ITEM_SUB_SEQ"] = self.MASK_ITEM_SUB_SEQ
        config["MASK_ITEM_SUB_SEQ_LEN"] = self.MASK_ITEM_SUB_SEQ_LEN
        self.config = config

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]

        device = item_seq.device
        n_items = dataset.num(self.ITEM_ID)

        time_seq_shifted = torch.roll(time_seq, 1)
        time_seq_shifted[:, 0] = time_seq[:, 0]
        time_delta = torch.abs(time_seq - time_seq_shifted)
        sub_seq_mask = torch.cumsum((time_delta > self.config["sub_time_delta"]), dim=1)

        all_subseqs = []
        for seq, length, mask in zip(item_seq, item_seq_len, sub_seq_mask):
            subseqs = mask[:length].unique()
            for session in subseqs:
                subseq_index = (mask == session).nonzero(as_tuple=True)[0]
                subseq = seq[subseq_index]
                if len(subseq) > 1:  # only sessions with more than 1 item
                    all_subseqs.append(subseq)

        # catch test forward
        if len(all_subseqs) == 0:
            all_subseqs = [torch.zeros((1, 50), dtype=torch.long)]

        item_sequence = torch.nn.utils.rnn.pad_sequence(all_subseqs, batch_first=True, padding_value=0.0)
        item_sequence_len = torch.count_nonzero(item_sequence, dim=1)

        all_subseqs_list = item_sequence.cpu().numpy().tolist()
        batch_size = len(all_subseqs)

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        for instance in all_subseqs_list:#sequence_instances:
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if item == 0:
                    break
                prob = random.random()
                if prob < self.mask_ratio and any(i != 0 for i in masked_sequence[:index_id]):
                    masked_sequence[index_id] = 0

            masked_item_sequence.append(masked_sequence)

        masked_item_sequence = torch.tensor(
            masked_item_sequence, dtype=torch.long, device=device
        ).view(batch_size, -1)

        #masked_item_sequence_lengths = torch.count_nonzero(masked_item_sequence, dim=1)

        new_dict = {
            self.ITEM_SUB_SEQ: item_sequence,
            self.ITEM_SUB_SEQ_LEN: item_sequence_len,
            self.MASK_ITEM_SUB_SEQ: masked_item_sequence,
            self.MASK_ITEM_SUB_SEQ_LEN: item_sequence_len,
        }
        interaction.update(Interaction(new_dict))

        return interaction


class InverseItemSequence:
    """
    inverse the seq_item, like this
        [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.INVERSE_ITEM_SEQ = "Inverse_" + self.ITEM_SEQ
        config["INVERSE_ITEM_SEQ"] = self.INVERSE_ITEM_SEQ

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        item_seq = item_seq.cpu().numpy()
        item_seq_len = item_seq_len.cpu().numpy()
        new_item_seq = []
        for items, length in zip(item_seq, item_seq_len):
            item = list(items[:length])
            zeros = list(items[length:])
            seqs = zeros + item
            new_item_seq.append(seqs)
        inverse_item_seq = torch.tensor(new_item_seq, dtype=torch.long, device=device)
        new_dict = {self.INVERSE_ITEM_SEQ: inverse_item_seq}
        interaction.update(Interaction(new_dict))
        return interaction


class CropItemSequence:
    """
    Random crop for item sequence.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.CROP_ITEM_SEQ = "Crop_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.CROP_ITEM_SEQ_LEN = self.CROP_ITEM_SEQ + self.ITEM_SEQ_LEN
        self.crop_eta = config["eta"]
        config["CROP_ITEM_SEQ"] = self.CROP_ITEM_SEQ
        config["CROP_ITEM_SEQ_LEN"] = self.CROP_ITEM_SEQ_LEN

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        crop_item_seq_list, crop_item_seqlen_list = [], []

        for seq, length in zip(item_seq, item_seq_len):
            crop_len = math.floor(length * self.crop_eta)
            crop_begin = random.randint(0, length - crop_len)
            crop_item_seq = np.zeros(seq.shape[0])
            if crop_begin + crop_len < seq.shape[0]:
                crop_item_seq[:crop_len] = seq[crop_begin : crop_begin + crop_len]
            else:
                crop_item_seq[:crop_len] = seq[crop_begin:]
            crop_item_seq_list.append(
                torch.tensor(crop_item_seq, dtype=torch.long, device=device)
            )
            crop_item_seqlen_list.append(
                torch.tensor(crop_len, dtype=torch.long, device=device)
            )
        new_dict = {
            self.CROP_ITEM_SEQ: torch.stack(crop_item_seq_list),
            self.CROP_ITEM_SEQ_LEN: torch.stack(crop_item_seqlen_list),
        }
        interaction.update(Interaction(new_dict))
        return interaction


class ReorderItemSequence:
    """
    Reorder operation for item sequence.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.REORDER_ITEM_SEQ = "Reorder_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.reorder_beta = config["beta"]
        config["REORDER_ITEM_SEQ"] = self.REORDER_ITEM_SEQ

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        reorder_seq_list = []

        for seq, length in zip(item_seq, item_seq_len):
            reorder_len = math.floor(length * self.reorder_beta)
            reorder_begin = random.randint(0, length - reorder_len)
            reorder_item_seq = seq.cpu().detach().numpy().copy()

            shuffle_index = list(range(reorder_begin, reorder_begin + reorder_len))
            random.shuffle(shuffle_index)
            reorder_item_seq[
                reorder_begin : reorder_begin + reorder_len
            ] = reorder_item_seq[shuffle_index]

            reorder_seq_list.append(
                torch.tensor(reorder_item_seq, dtype=torch.long, device=device)
            )
        new_dict = {self.REORDER_ITEM_SEQ: torch.stack(reorder_seq_list)}
        interaction.update(Interaction(new_dict))
        return interaction


class UserDefinedTransform:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        pass
