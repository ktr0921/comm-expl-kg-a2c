# *=*: {triple} exist\nlooking for {head_j}
# -=-: cskg found, {head2tail_n_score[head_j]}

# =-=: {head_j} NOT found in {f_read}
# =*=: {triple} does not exist

import os
import sys
import re
import random
import time
import spacy
import itertools
import numpy as np
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from models.components import EdgePredictor
from utils.representations import pad_sequences

# atomic 2020 edge
pyhsical_entity = [
    'ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty', 'CapableOf', 'Desires', 'NotDesires'
]
event_centered = [
    'isAfter', 'HasSubEvent', 'IsBefore', 'HinderedBy', 'Causes', 'xReason', 'isFilledBy',
]
social_interaction = [
    'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant',
]
atomic_2020_edge = pyhsical_entity + event_centered + social_interaction

# edge to readable phrase
edge_to_readable_phrase = {
    'xNeed': 'PersonX needs',
    'xEffect': 'PersonX',
    'xIntent': 'PersonX wants',
    'xWant': 'PersonX wants',
    'xReact': 'PersonX is',
    'xAttr': 'PersonX is',
}


class EdgeSampling:
    def __init__(self, edge, edge_sampling_n, edge_sampling_type, sp, batch_size, params=None):
        # edge
        if edge == 'all':
            self.edge = event_centered + social_interaction
        elif edge == 'event_centered':
            self.edge = event_centered
        elif edge == 'social_interaction':
            self.edge = social_interaction
        else:
            edge = {edge.strip('[] ') for edge in edge.split(',')}
            assert all(a_i in atomic_2020_edge for a_i in edge)
            self.edge = [a_i for a_i in atomic_2020_edge if a_i in edge]

        # if predict
        if 'predict' in edge_sampling_type:
            self.id_edge = list(edge)
            self.edge_id = {e: i for i, e in enumerate(self.id_edge)}
            self.edge_predictor = EdgePredictor(len(sp), batch_size, len(self.id_edge))
        else:
            self.id_edge = None
            self.edge_id = None
            self.edge_predictor = None

        self.edge_sampling_n = edge_sampling_n
        self.edge_sampling_type = edge_sampling_type
        self.sp = sp

        self.node2dist = dict()

    def get_node_id(self, node, root_node_person2you=None):
        '''
        node: all the nodes for a single hop, like 0-th, 1-st, ...; list of str => [...]
        root_node_person2you: dictionary from 'root_node' to 'root_node' replaced 'PersonX' with 'You'; dict of str =>
            {root_node: root_node_replace_PersonX_You}

        node_id: numpy array that transformed 'node' to token id; numpy.array => [...node..., maxlen=300]
        '''
        # change node from 'PersonX' to 'You' & get node id
        node_change = {
            'PersonX needs': 'you need', 'PersonX wants': 'you want',
            'PersonX is': 'you are', 'Is PersonX': 'Are you', 'is PersonX': 'are you',
            'PersonX does': 'you do', 'Does PersonX': 'Do you', 'does PersonX': 'do you',
            'PersonX': 'you', "PersonX's": 'your',
        }
        node_id = []
        for node_i in node:
            node_i = root_node_person2you[node_i] if node_i in root_node_person2you.keys() else node_i
            for k, v in node_change.items():
                node_i = node_i.replace(k, v)
            node_i = self.sp.encode_as_ids(node_i[0].upper() + node_i[1:])
            node_id.append(node_i)
        node_id = pad_sequences(node_id, maxlen=300)
        return node_id

    def get_edge(self, node, root_node_person2you=None, h_in=None):
        '''
        node: all the nodes for a single hop, like 0-th, 1-st, ...; list of str => [...]
        root_node_person2you: dictionary from 'root_node' to 'root_node' replaced 'PersonX' with 'You'; dict of str =>
            {root_node: root_node_replace_PersonX_You}
        h_in: previous hidden state, used for only predict

        edge: sampled edges (list of str) for each node (list); list of list of str => [[...], [...], ...node...]

        depending on 'self.edge_sampling_type', either choose all, randomly sample, predict based on 'node' phrase
        '''
        self.node2dist = dict()
        if self.edge_sampling_type == 'all':
            edge = [self.edge] * len(node)
        elif self.edge_sampling_type == 'random':  # random_no_rep
            edge = [random.sample(self.edge, k=self.edge_sampling_n) for _ in range(len(node))]
            # edge = [random.choices(list(edge), k=edge_sampling_n) for _ in range(len(node))]
        elif self.edge_sampling_type == 'predict':  # predict_no_rep
            node_id = self.get_node_id(node, root_node_person2you)
            edge_distribution, h_out = self.edge_predictor(node_id, h_in)
            dist = F.softmax(edge_distribution.to(torch.float64), dim=-1).to(torch.float64)
            edge_id = (dist + torch.tensor(1e-5)).multinomial(num_samples=self.edge_sampling_n)
            # edge_id = (dist + torch.tensor(1e-5)).multinomial(num_samples=edge_sampling_n, replacement=True)
            edge = [[self.id_edge[_ij.item()] for _ij in _i] for _i in edge_id]

            # this is for optim, which is used in later stage
            self.node2dist = dict(zip(node, dist))
        else:
            edge = []
        return edge


class CSKGConstruction:
    def __init__(self, comet, cskg_size, num_generate, store_comet_output,
                 vv_edge, vv_edge_sampling_n, vv_edge_sampling_type,
                 sp, batch_size, params=None):
        # comet
        self.COMeT = comet
        self.cskg_size = cskg_size
        self.num_generate = num_generate
        self.store_comet_output = store_comet_output

        # node2node
        self.Node2NodeEdgeSampling = EdgeSampling(
            vv_edge, vv_edge_sampling_n, vv_edge_sampling_type,
            sp, batch_size, params=params
        )

        self.params = params

    def get_init_cskg(self, root_node):
        '''
        root_node: the previous action and the current observation for all the environments; list of str => [batch_size]

        cskg_df: the initial cskg; dataframe of ['batch_group', 'node0', 'score0']

        the same element in 'root_node' (a.k.a different environments/batches) is concatenated as 'node0' and
        'batch_group' indicates which batch it belongs & 'score0'=0 for all
        '''
        # initialize root_node & create batch_group
        heads, batch_heads = root_node, dict()
        for i in range(len(heads)):
            batch_heads[heads[i]] = batch_heads[heads[i]] + [str(i)] if heads[i] in batch_heads.keys() else [str(i)]
        (k, v) = zip(*batch_heads.items())

        # create empty cskg_df
        cskg_df = pd.DataFrame(
            zip([','.join(sorted(v_i)) for v_i in v], k, [0] * len(batch_heads)),
            columns=['batch_group', 'node0', 'score0']
        )
        return cskg_df

    def search_triple(self, head_j, vv_edge_jk, query, query_triple, cskg_i_values):
        '''
        head_j: head node; str
        vv_edge_jk: edge; str
        query: query input to self.COMeT.generate_with_scores; list of str => [...]
        query_triple: the directory that the query is going to be saved; list of str => [...]
        cskg_i_values: values in cskg_df that if the triple exist, store the triple so that we do not have to query it

        query: query input to self.COMeT.generate_with_scores; list of str => [...]
        query_triple: the directory that the query is going to be saved; list of str => [...]
        cskg_i_values: values in cskg_df that if the triple exist, store the triple so that we do not have to query it

        search the file, f'{self.store_comet_output}/cskg/{self.params["env_name"]}/{...head_j...}/{vv_edge_jk}/{self.num_generate}'
            if exist, read it and create dictionary for each head
            search the head, 'head_j'
                if exist, read it and store it to 'cskg_i_values'
                else, append to query_triple and query
            else, append to query_triple and query
        '''
        # get search file and create parent directory
        head_tmp = head_j.replace(".", " ").split()[::3]
        k = int((len(head_tmp) - 1) / 2) if len(head_tmp) < 10 else 5
        triple = f'{self.store_comet_output}/cskg/{self.params["env_name"]}/{"".join(head_tmp[:k] + head_tmp[-k:])}/{vv_edge_jk}/{self.num_generate}'
        if not os.path.exists(triple.rsplit('/', 1)[0]):
            try:
                os.makedirs(triple.rsplit('/', 1)[0])
            except OSError as e:
                print(e)

        if os.path.exists(triple):  # if the file exist, read it and create dictionary for head
            with open(triple, 'r') as f:
                f_read = f.read().strip()

            head2tail_n_score = dict()
            for f_j in f_read.strip().split('\n\n'):
                if f_j.strip():
                    tmp = f_j.rsplit('\n', 1)
                    if len(tmp) == 2:
                        head2tail_n_score[tmp[0]] = tmp[1]
                    else:
                        print(f'ERROR: cskg. remove "{tmp}" in "{triple}".\nCondition: len(tmp) == 2')

            if head_j in head2tail_n_score.keys():  # if 'head_j' exist, read it and store it to 'cskg_i_values'
                for tail_n_score in head2tail_n_score[head_j].strip().split('\t\t'):
                    tail_jk, score_jk = tail_n_score.rsplit('\t', 1)
                    cskg_i_values.append([head_j, vv_edge_jk, tail_jk, float(score_jk)])

            else:  # else, append to query_triple and query
                query_triple.append(triple)
                query.append(f'{head_j} {vv_edge_jk} [GEN]')

        else:  # else, append to query_triple and query
            query_triple.append(triple)
            query.append(f'{head_j} {vv_edge_jk} [GEN]')

        return query, query_triple, cskg_i_values

    def get_cskg(self, root_node, root_node_person2you):
        '''
        root_node: the previous action and the current observation for all the environments; list of str => [batch_size]
        root_node_person2you: dictionary from 'root_node' to 'root_node' replaced 'PersonX' with 'You'; dict of str =>
            {root_node: root_node_replace_PersonX_You}

        cskg_df: cskg; dataframe of ['batch_group', 'node0', 'score0', ('edge1', 'node1', 'score1'), ('edge2', ...]
        '''
        # get init cskg
        cskg_df = self.get_init_cskg(root_node)

        # expand cskg
        for i in range(self.cskg_size):
            # init query & init head & get node2node edge
            cskg_i_values, query, query_triple, head = [], [], [], cskg_df.loc[:, f'node{i}']
            vv_edge = self.Node2NodeEdgeSampling.get_edge(head, root_node_person2you)

            # create query with head & edge or read triples
            for head_j, vv_edge_j in zip(head, vv_edge):
                for vv_edge_jk in vv_edge_j:
                    if self.store_comet_output is not None:
                        query, query_triple, cskg_i_values = self.search_triple(
                            head_j, vv_edge_jk, query, query_triple, cskg_i_values
                        )
                    else:
                        query.append(f'{head_j} {vv_edge_jk} [GEN]')

            # get tails and scores given query
            tail, score = self.COMeT.generate_with_scores(
                query, num_generate=self.num_generate, min_length=0, max_length=24
            )

            # append tails and scores to cskg dataframe, and if self.store_comet_output is not None, write it
            for j, v in enumerate(zip(query, tail, score)):
                query_j, tail_j, score_j = v
                head_j, edge_j, _ = query_j.rsplit(' ', 2)

                save_tail_score = []
                for tail_jk, score_jk in zip(tail_j, score_j):
                    tail_jk = tail_jk.strip()

                    # pre-process tail
                    if tail_jk != 'none':
                        tail_filter = [
                            'you ', 'person ', 'personx ', 'persony ', 'personz ', 'person x ', 'person y ', 'person z '
                        ]
                        for _j in tail_filter:
                            tail_jk = tail_jk[len(_j):] if _j == tail_jk[:len(_j)].lower() else tail_jk
                        tail_jk = f'{edge_to_readable_phrase[edge_j]} {tail_jk}'
                        tail_jk = tail_jk.strip()
                        tail_jk = f'{tail_jk}.' if (tail_jk[-1].isalpha()) or (tail_jk[-1].isdigit()) else tail_jk
                        tail_jk = re.sub(r'[ ]*\.[\. ]*', '. ', re.sub(r'\n+', '.', tail_jk))
                        tail_jk = ' '.join(tail_jk.split())

                    # append values to 'cskg_i_values'
                    cskg_i_values.append([head_j, edge_j, tail_jk, score_jk.item()])
                    save_tail_score += [f'{tail_jk.strip()}\t{score_jk.item()}']

                with open(query_triple[j], 'a') as f:
                    f.write('\n\n' + head_j + '\n' + '\t\t'.join(save_tail_score))

            # create dataframe 'cskg_i' and merge it with 'cskg_df'
            columns = [f'node{i}', f'edge{i+1}', f'node{i+1}', f'score{i+1}']
            cskg_i = pd.DataFrame(cskg_i_values, columns=columns)
            cskg_df = pd.merge(cskg_df, cskg_i, how='outer', on=f'node{i}')

        return cskg_df
