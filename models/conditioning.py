# _=_: va score from {head_j} to {act_i} found, {act2score[act_i]}
# =_=: va score from {head_j} to {act_i} NOT found in {triple}

# =0=: {head_j} NOT found in {triple}
# =1=: {triple} NOT found

import os
import sys
import re
import random
import time
import spacy
import itertools
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from models.cskg_construction import EdgeSampling
from models.comet import trim_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pyhsical_entity = [
    'ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty', 'CapableOf', 'Desires', 'NotDesires'
]
event_centered = [
    'isAfter', 'HasSubEvent', 'IsBefore', 'HinderedBy', 'Causes', 'xReason', 'isFilledBy',
]
social_interaction = [
    'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant',
]
direction_tmpl = {
    'n': 'north', 's': 'south', 'w': 'west', 'e': 'east',
    'north': 'north', 'south': 'south', 'west': 'west', 'east': 'east',
    'nw': 'northwest', 'sw': 'southwest', 'ne': 'northeast', 'se': 'southeast',
    'northwest': 'northwest', 'southwest': 'southwest', 'northeast': 'northeast', 'southeast': 'southeast',
    'u': 'up', 'd': 'down', 'up': 'up', 'down': 'down',
    'front': 'front', 'back': 'back', 'right': 'right', 'left': 'left',
    'in': 'in', 'out': 'out',
}
edge_to_readable_phrase = {
    'xNeed': 'PersonX needs',
    'xEffect': 'PersonX',
    'xIntent': 'PersonX wants',
    'xWant': 'PersonX wants',
    'xReact': 'PersonX is',
    'xAttr': 'PersonX is',
}
atomic_2020_edge = pyhsical_entity + event_centered + social_interaction


class COMeTExploration:
    def __init__(self, comet, cskg_size, act_sampling_type,
                 va_edge, va_edge_sampling_n, va_edge_sampling_type, sp,
                 batch_size, vocab_id2tmpl, vocab_id2tkn,
                 vv_prev_score_gamma, vv_score_gamma, va_score_gamma, tmpl_score_gamma, aa_score_gamma,
                 params=None):
        super().__init__()

        # comet
        self.COMeT = comet
        self.cskg_size = cskg_size  # delete if can
        self.act_sampling_type = act_sampling_type

        # node2act
        self.Node2ActEdgeSampling = EdgeSampling(
            va_edge, va_edge_sampling_n, va_edge_sampling_type,
            sp, batch_size, params=params
        )

        # agent
        self.batch_size = batch_size
        self.vocab_id2tmpl = vocab_id2tmpl
        self.vocab_id2tkn = vocab_id2tkn

        # gamma
        self.vv_prev_score_gamma = vv_prev_score_gamma
        self.vv_score_gamma = vv_score_gamma
        self.va_score_gamma = va_score_gamma
        self.tmpl_score_gamma = tmpl_score_gamma
        self.aa_score_gamma = aa_score_gamma

        self.params = params
        self.store_comet_output = params['store_comet_output']

    def get_vv_score(self, cskg_df):
        '''
        cskg_df: cskg; dataframe of ['batch_group', 'node0', 'score0', ('edge1', 'node1', 'score1'), ('edge2', ...]

        vv_score_df: node2node score; dataframe of ['batch_group', 'head', 'edge', 'tail', 'node2node_score']

        get accumulated score; cskg => ['batch_group', 'node0', 'score0', 'score0-acc', ('edge1', 'node1', 'score1', 'score1-acc'), ...]
        get node2node score at node0; vv_score_df with 'node0', ['batch_group', -, -, 'node0', 0]
        get node2node score at all the nodes; vv_score_df with 'node0',
            ['batch_group', 'node0', 'edge1', 'node1', 'score1-acc'] & ['batch_group', 'node1', 'edge2', 'node2', 'score2-acc'] & ...
            exclude those 'tail' that are not from 'none' or are not 'none' [..., 'none', ..., ..., ...] & [..., ..., ..., 'none', ...]
        convert data and drop duplicates; 'node2node_score' column to numeric & drop duplicates
        '''
        # get node2node score at node0
        node0 = [[batch_i, '-', '-', node0_i, 0] for batch_i,node0_i in cskg_df.loc[:, ['batch_group', 'node0']].values]
        columns = ['batch_group', 'head', 'edge', 'tail', 'node2node_score']
        vv_score_df = pd.DataFrame(node0, columns=columns)

        # get node2node score at all the nodes that are not from 'none' or are not 'none'
        cskg_df.loc[:, 'score0-acc'] = 0
        for i in range(self.cskg_size):
            cskg_df.loc[:, f'score{i+1}-acc'] = cskg_df.loc[:, f'score{i}-acc'] * self.vv_prev_score_gamma \
                                                + cskg_df.loc[:, f'score{i+1}']
            in_columns = ['batch_group', f'node{i}', f'edge{i+1}', f'node{i+1}', f'score{i+1}-acc']
            vv_i_df = cskg_df.loc[:, in_columns]
            vv_i_df.columns = columns
            idx = (vv_i_df.loc[:, 'head'] != 'none') & (vv_i_df.loc[:, 'tail'] != 'none')
            vv_score_df = vv_score_df.append(vv_i_df.loc[idx, :], ignore_index=True)

        # convert data and drop duplicates
        vv_score_df.loc[:, 'node2node_score'] = pd.to_numeric(vv_score_df.loc[:, 'node2node_score'])
        vv_score_df = vv_score_df.drop_duplicates(ignore_index=True)

        return vv_score_df

    def get_aa_score(self, tmpl_token_id, tmpl_dist, obj1_token_id, obj1_dist, obj2_token_id, obj2_dist, graph, tmpl_token_id_batch):
        '''
        act_name: action names, [tmpl], [tmpl, obj1], or [tmpl, obj1, obj2]
        softmax_act_dist: corresponding softmax distribution
        act_token_id: corresponding sampled token id

        aa_score_df: agt2act score; dataframe of ['batch', (f'tmpl_token', f'tmpl_token_id', f'agt2tmpl_score'),
            (f'obj1_token', f'obj1_token_id', f'agt2obj1_score'), ..., 'act4comet', 'agt2act_score']

        get agt2act score for all the tmpl, obj1, obj2. this is general algorithm, so it can be used not only for tmpl,
            but also obj, so obtain ['batch', (f'tmpl_token', f'tmpl_token_id', f'agt2tmpl_score'), (f'obj1_token',
            f'obj1_token_id', f'agt2obj1_score'), ...]
        get input to comet and agt2act score; input to comet is (take OBJ from OBJ + pencil + table = take pencil from
            table) or (east = go east); agt2act score is accumulated 'agt2tmpl_score', 'agt2obj1_score', ... score
        '''
        tmpl_token_id_, obj1_token_id_, obj2_token_id_ = \
            tmpl_token_id.squeeze(-1), obj1_token_id.squeeze(-1), obj2_token_id.squeeze(-1)
        aa_score_value, batch_number = [], 0
        for i in range(tmpl_token_id_.size(0)):
            softmax_tmpl_dist_i = F.softmax(tmpl_dist[i], dim=0)
            softmax_obj1_dist_i_ = F.softmax(obj1_dist[i][graph[i]], dim=0)
            softmax_obj1_dist_i = torch.zeros([graph.size(-1)]).to(device)
            softmax_obj1_dist_i[graph[i]] += softmax_obj1_dist_i_.detach().clone()
            softmax_obj2_dist_i_ = F.softmax(obj2_dist[i][graph[i]], dim=0)
            softmax_obj2_dist_i = torch.zeros([graph.size(-1)]).to(device)
            softmax_obj2_dist_i[graph[i]] += softmax_obj2_dist_i_.detach().clone()

            aa_score_value_i = (
                batch_number, tmpl_token_id_[i].item(), self.vocab_id2tmpl[tmpl_token_id_[i].item()],
                (softmax_tmpl_dist_i[tmpl_token_id_[i].item()] + torch.tensor(1e-7)).log().item(),
                obj1_token_id_[i].item(), self.vocab_id2tkn[obj1_token_id_[i].item()],
                (softmax_obj1_dist_i[obj1_token_id_[i].item()] + torch.tensor(1e-7)).log().item(),
                obj2_token_id_[i].item(), self.vocab_id2tkn[obj2_token_id_[i].item()],
                (softmax_obj2_dist_i[obj2_token_id_[i].item()] + torch.tensor(1e-7)).log().item(),
            )
            aa_score_value.append(aa_score_value_i)
            if i == sum(tmpl_token_id_batch[:batch_number+1])-1:
                batch_number += 1

        columns = [
            'batch', 'tmpl_token_id', 'tmpl_token', 'agt2tmpl_score',
            'obj1_token_id', 'obj1_token', 'agt2obj1_score',
            'obj2_token_id', 'obj2_token', 'agt2obj2_score',
        ]
        aa_score_df = pd.DataFrame(aa_score_value, columns=columns)

        # get input to comet and agt2act score
        def f_act4comet(x):  # construct act4comet
            act4comet = 'go ' + direction_tmpl[x.loc['tmpl_token'].lower()] \
                if x.loc['tmpl_token'].lower() in set(direction_tmpl.keys()) else x.loc['tmpl_token']
            act4comet = act4comet.replace('OBJ', x.loc['obj1_token'], 1) if 'obj1_token' in x.index else act4comet
            act4comet = act4comet.replace('OBJ', x.loc['obj2_token'], 1) if 'obj2_token' in x.index else act4comet
            return act4comet.split('OBJ')[0].strip()

        def f_agt2act_score(x):  # construct agt2act score
            n = x.loc['tmpl_token'].count('OBJ')
            agt2tmpl_score = -10 if x.loc['agt2tmpl_score'] < -10 else x.loc['agt2tmpl_score']
            gamma_agt2tmpl_score = self.tmpl_score_gamma * agt2tmpl_score
            if (n == 2) and ('agt2obj1_score' in x.index) and ('agt2obj2_score' in x.index):
                return (gamma_agt2tmpl_score + x.loc['agt2obj1_score'] + x.loc['agt2obj2_score']) / 3
            elif (n == 1) and ('agt2obj1_score' in x.index):
                return (gamma_agt2tmpl_score + x.loc['agt2obj1_score']) / 2
            elif (n == 0):
                return gamma_agt2tmpl_score
            else:
                import sys
                sys.exit('tmpl_token has OBJ more than 2')

        aa_score_df.loc[:, 'act4comet'] = aa_score_df.apply(f_act4comet, axis=1)
        aa_score_df.loc[:, 'agt2act_score'] = aa_score_df.apply(f_agt2act_score, axis=1)

        # return act_k_name, aa_score_df
        return aa_score_df

    def search_va_score(self, node_edge, act):
        '''
        node_edge: node + edge combined for COMET input
        act: act for COMET output

        va_score_df: node2act score; dataframe of ['batch_group', 'node', 'edge4act', 'act', 'node2act_score']

        store triples to generate in triple_missing & store data if triple exist
        get search file f'{self.store_comet_output}/cskg/{self.params["env_name"]}/{...head_j...}/{va_edge_jk}/va' and create parent directory
        if the file exist, read it and create dictionary for head
            if head_j exist, create dictionary for act
                for each act, if act exist, store False in triple_missing & store data in rows
            else, store True in triple_missing (this means generate the data)
        else, store True in triple_missing for every act (this means generate the data)
        transpose triple_missing from [node, act] => [act, node], so that it can be iterated with 'act'
        '''
        # store triples to generate in triple_missing & store data if triple exist
        triple_missing, rows = [], []
        for j, head_j in enumerate(node_edge):
            # get search file and create parent directory
            head_j, va_edge_jk, _ = head_j.rsplit(' ', 2)
            head_tmp = head_j.replace(".", " ").split()[::3]
            k = int((len(head_tmp) - 1) / 2) if len(head_tmp) < 10 else 5
            triple = f'{self.store_comet_output}/cskg/{self.params["env_name"]}/{"".join(head_tmp[:k] + head_tmp[-k:])}/{va_edge_jk}/va'
            if not os.path.exists(triple.rsplit('/', 1)[0]):
                try:
                    os.makedirs(triple.rsplit('/', 1)[0])
                except OSError as e:
                    print(e)

            triple_missing_j = []
            if os.path.exists(triple):  # if the file exist, read it and create dictionary for head
                with open(triple, 'r') as f:
                    f_read = f.read()

                head2act_score = dict()
                for f_j in f_read.strip().split('\n\n'):
                    if f_j.strip():
                        tmp = f_j.split('\n', 1)
                        if len(tmp) == 2:
                            if tmp[0] in head2act_score.keys():
                                head2act_score[tmp[0]] += '\t\t' + tmp[1].strip()
                            else:
                                head2act_score[tmp[0]] = tmp[1].strip()
                        else:
                            print(f'ERROR: va. remove "{tmp}" in "{triple}".\nCondition: len(tmp) == 2')

                act2score = dict()
                if head_j in head2act_score.keys():  # if head_j exist, create dictionary for act
                    for f_jk in head2act_score[head_j].strip().split('\t\t'):
                        if f_jk.strip():
                            tmp = f_jk.split('\t')
                            if (len(tmp) == 2) and (tmp[0].split()[0] == '</s>') \
                                    and (tmp[1].count('-') < 2) and (tmp[1].count('.') < 2) \
                                    and (tmp[1].replace('-', '').replace('.', '').isdigit()):
                                act2score[tmp[0]] = tmp[1]
                            else:
                                print(f'ERROR: va. remove "{f_jk}" in "{head_j}" in "{triple}".\n'
                                      + 'Condition is (len(tmp) == 2) and (tmp[0].split()[0] == "</s>") '
                                      + 'and (tmp[1].count("-") < 2) and (tmp[1].count(".") < 2) and '
                                      + '(tmp[1].replace("-", "").replace(".", "").isdigit())')

                    for act_i in act:  # for each act, if act exist, store False in triple_missing & store data in rows
                        if act_i in act2score.keys():
                            triple_missing_j.append(False)
                            act2score_value = act2score[act_i]
                            act2score_value = re.sub('[-]+', '-', act2score_value)
                            act2score_value = re.sub('\.+', '.', act2score_value)
                            act2score_value = '-'.join(act2score_value.split('-')[:2]) \
                                if act2score_value.count('-') > 1 else act2score_value
                            act2score_value = '.'.join(act2score_value.split('.')[:2]) \
                                if act2score_value.count('.') > 1 else act2score_value
                            rows.append([head_j, va_edge_jk, act_i.split(' ', 2)[-1], float(act2score_value)])

                        else:  # else, store True in triple_missing (this means generate the data)
                            triple_missing_j.append(True)

                else:  # else, store True in triple_missing for every act (this means generate the data)
                    triple_missing_j = [True] * len(act)

            else:  # else, store True in triple_missing for every act (this means generate the data)
                triple_missing_j = [True] * len(act)

            triple_missing.append(triple_missing_j)
        # transpose triple_missing from [node, act] => [act, node], so that it can be iterated with 'act'
        triple_missing = list(map(list, zip(*triple_missing)))

        return triple_missing, rows

    def generate_va_score(self, node_edge, act, n_square=True, ):
        '''
        node_edge: node + edge combined for COMET input
        act: act for COMET output
        n_square: whether use n**2 for node2act score

        va_score_df: node2act score; dataframe of ['batch_group', 'node', 'edge4act', 'act', 'node2act_score']

        get node & edge; seperate node and edge for creating node2act score
        if xEffect & oEffect, skip ('</s> PersonX', [    2, 18404,  1000]), otherwise skip ('</s> to', [2, 7])
        tokenize node_edge & search node2act score
        '''
        with torch.no_grad():
            # get node & edge
            node2rows, edge2rows = [], []
            for node_edge_i in node_edge:
                node2rows_i, edge2rows_i, _ = node_edge_i.rsplit(' ', 2)
                node2rows.append(node2rows_i)
                edge2rows.append(edge2rows_i)
            # if xEffect & oEffect, skip ('</s> PersonX', [    2, 18404,  1000]), otherwise skip ('</s> to', [2, 7])
            start = 3 if len(set(edge2rows) - {'xEffect', 'oEffect'}) <= 0 else 2

            # tokenize node_edge & search node2act score
            node_edge_token = self.COMeT.tokenizer(node_edge, return_tensors="pt", truncation=True, padding="max_length")
            input_ids, att_mask = trim_batch(**node_edge_token.to(device), pad_token_id=self.COMeT.tokenizer.pad_token_id)
            triple_missing, rows = self.search_va_score(node_edge, act) \
                if self.store_comet_output is not None else ([[True] * len(node_edge)] * len(act), [])

            # get node2act score
            va_score_df = pd.DataFrame(rows, columns=['node', 'edge4act', 'act', 'node2act_score'])
            for act_i, triple_missing_i in zip(act, triple_missing):
                # tokenize act
                output_ids = self.COMeT.tokenizer(act_i, return_tensors="pt", add_special_tokens=False).input_ids
                output_ids = output_ids.to(device)

                # iterate to get log likelihood of each token after '</s> PersonX' or '</s> to'
                trg_token_log_prob = []
                if sum(triple_missing_i) != 0:
                    for i in range(start, output_ids.size(1)):
                        generated_outputs = self.COMeT.model.generate(
                            input_ids=input_ids[triple_missing_i],
                            attention_mask=att_mask[triple_missing_i],
                            decoder_start_token_id=output_ids[:, :i],
                            max_length=i + 1,
                            num_beams=1,
                            num_return_sequences=1,
                            output_scores=True,
                            return_dict_in_generate=True,
                            do_sample=False,
                        )

                        # get log likelihood
                        next_token_dist = generated_outputs.scores[0].softmax(-1)
                        trg_token_log_prob_i = (next_token_dist[:, output_ids[:, i].item()] + torch.tensor(1e-7)).log()
                        trg_token_log_prob.append(trg_token_log_prob_i)

                    # calculate avg lod likelihood
                    trg_token_log_prob = torch.stack(trg_token_log_prob).sum(0) / (len(trg_token_log_prob) ** 2) \
                        if n_square else torch.stack(trg_token_log_prob).sum(0) / len(trg_token_log_prob)

                    if self.store_comet_output is not None:
                        node2rows_j = [node2rows[j] for j, _ij in enumerate(triple_missing_i) if _ij]
                        edge2rows_j = [edge2rows[j] for j, _ij in enumerate(triple_missing_i) if _ij]
                        for head_j, va_edge_jk, log_prob_i in zip(node2rows_j, edge2rows_j, trg_token_log_prob.tolist()):
                            head_tmp = head_j.replace(".", " ").split()[::3]
                            k = int((len(head_tmp) - 1) / 2) if len(head_tmp) < 10 else 5
                            triple = f'{self.store_comet_output}/cskg/{self.params["env_name"]}/{"".join(head_tmp[:k] + head_tmp[-k:])}/{va_edge_jk}/va'

                            with open(triple, 'a') as f:
                                f.write(f'\n\n{head_j}\n{act_i}\t{log_prob_i}')

                    else:
                        node2rows_j = node2rows
                        edge2rows_j = edge2rows

                    # append the results
                    rows = list(zip(
                        node2rows_j, edge2rows_j, [' '.join(act_i.split()[2:])] * len(node2rows_j), trg_token_log_prob.tolist()
                    ))
                    rows_df = pd.DataFrame(rows, columns=va_score_df.columns)
                    va_score_df = va_score_df.append(rows_df, ignore_index=True)

        return va_score_df

    def get_va_score(self, vv_score_df, aa_score_df, root_node_person2you):
        '''
        vv_score_df: node2node score; dataframe of ['batch_group', 'head', 'edge', 'tail', 'node2node_score']
        aa_score_df: agt2act score; dataframe of ['batch', (f'tmpl_token', f'tmpl_token_id', f'agt2tmpl_score'),
            (f'obj1_token', f'obj1_token_id', f'agt2obj1_score'), ..., 'act4comet', 'agt2act_score']
        root_node_person2you: dictionary from 'root_node' to 'root_node' replaced 'PersonX' with 'You'; dict of str =>
            {root_node: root_node_replace_PersonX_You}

        va_score_df: node2act score; dataframe of ['batch_group', 'node', 'edge4act', 'act', 'node2act_score']

        initialize query (node_personx, node_to) & head (node_tmp) & get node2act edge
        create query with head & relation. separate two queries (node_personx, node_to); 'head', 'tail' of node2node
            score + edge from EdgeSampling.get_edge()
        get action; 'act4comet' of agt2act score
        get node2act_score_df for node_personx (xEffect or oEffect edge) & node_to (rest edges) to action
        add batch to node2act score; this divides 'batch_group' to each 'batch'
        '''
        va_score_df = pd.DataFrame(columns=['batch_group', 'node', 'edge4act', 'act', 'node2act_score'])
        for batch_group in set(vv_score_df.loc[:, 'batch_group']):
            # init query (node_personx, node_to) & head (node_tmp) & get node2act edge
            node_personx, node_to = [], []
            df_tmp = vv_score_df.loc[vv_score_df.loc[:, 'batch_group'] == batch_group, ['head', 'tail']]
            node_tmp = list(set(itertools.chain.from_iterable(df_tmp.values.tolist())) - {'-'})
            va_edge = self.Node2ActEdgeSampling.get_edge(node_tmp, root_node_person2you)

            # create query with head & relation
            for node_tmp_i, va_edge_i in zip(node_tmp, va_edge):
                # create node queries for COMeT. if xEffect or oEffect, node, otherwise node_to.
                # node is used for actions that removed ' to ' in front of it
                for va_edge_ij in va_edge_i:
                    if va_edge_ij[1:] == 'Effect':
                        node_personx.append(f'{node_tmp_i} {va_edge_ij} [GEN]')
                    else:
                        node_to.append(f'{node_tmp_i} {va_edge_ij} [GEN]')

            # get action
            aa_score_df_idx = aa_score_df.loc[:, 'batch'].isin([int(_i) for _i in batch_group.split(',')])
            act_all = set(aa_score_df.loc[aa_score_df_idx, 'act4comet'].to_list())

            # get node2act_score_df for node_personx (relation of xEffect or oEffect) to action
            if len(node_personx) > 0:
                act = {'</s> PersonX ' + act_i.strip() for act_i in act_all}
                va_score_df_i = self.generate_va_score(node_personx, act, n_square=False)
                va_score_df_i.loc[:, 'batch_group'] = batch_group
                va_score_df = va_score_df.append(va_score_df_i, ignore_index=True)

            # get node2act_score_df for node_to to action
            if len(node_to) > 0:
                act = {'</s> to ' + act_i.strip() for act_i in act_all}
                va_score_df_i = self.generate_va_score(node_to, act, n_square=False)
                va_score_df_i.loc[:, 'batch_group'] = batch_group
                va_score_df = va_score_df.append(va_score_df_i, ignore_index=True)

        # add batch to node2act score
        va_score_df_tmp = pd.DataFrame(columns=va_score_df.columns)
        for b in set(va_score_df.loc[:, 'batch_group']):
            va_score_df_tmp_i = va_score_df.loc[va_score_df.loc[:, 'batch_group'] == b, :]
            for batch_group_i in b.split(','):
                with pd.option_context('mode.chained_assignment', None):
                    va_score_df_tmp_i.loc[:, 'batch'] = int(batch_group_i)
                va_score_df_tmp = va_score_df_tmp.append(va_score_df_tmp_i, ignore_index=True)

        va_score_df = va_score_df_tmp
        va_score_df.loc[:, 'node2act_score'] = pd.to_numeric(va_score_df.loc[:, 'node2act_score'])
        va_score_df = va_score_df.rename(
            {'act': 'act4comet', 'node': 'tail'}, axis='columns'
        )

        return va_score_df

    def comet_conditioning(self, tmpl, obj1, obj2, graph, tmpl_token_id_batch, cskg_df, Node2NodeEdgeSampling, root_node_person2you):
        '''
        === [NODE2NODE] ===: get node2node score
        === [AGT2ACT] ===: get agt2act score
        === [NODE2ACT] ===: get node2act score
        === [AGT2ACT + NODE2ACT + NODE2NODE] ===: merge them
        === [TOTAL SCORE] ===: get total score
        '''

        tmpl_token_id, tmpl_dist, obj_num_in_tmpl = tmpl
        obj1_token_id, obj1_dist = obj1
        obj2_token_id, obj2_dist = obj2

        # === [NODE2NODE] ===
        vv_score_df = self.get_vv_score(cskg_df)

        # === [AGT2ACT] ===
        aa_score_df = self.get_aa_score(
            tmpl_token_id, tmpl_dist, obj1_token_id, obj1_dist, obj2_token_id, obj2_dist,
            graph, tmpl_token_id_batch
        )

        # === [NODE2ACT] ===
        va_score_df = self.get_va_score(vv_score_df, aa_score_df, root_node_person2you)

        # === [AGT2ACT + NODE2ACT + NODE2NODE] ===
        all_score_df = pd.merge(aa_score_df, va_score_df, how='left', on=['batch', 'act4comet'])
        all_score_df = pd.merge(all_score_df, vv_score_df, how='outer', on=['tail', 'batch_group'])

        # === [TOTAL SCORE] ===
        all_score_df.loc[:, 'total_score'] = self.vv_score_gamma * all_score_df.loc[:, 'node2node_score'] + \
                                             self.va_score_gamma * all_score_df.loc[:, 'node2act_score'] + \
                                             self.aa_score_gamma * all_score_df.loc[:, 'agt2act_score']

        # ['batch', f'{act_k_name}_token', f'{act_k_name}_token_id', f'agt2{act_k_name}_score'] act_k_name=tmpl,obj1,...
        act_columns = [
            'batch', 'batch_group', 'tmpl_token', 'tmpl_token_id', 'agt2tmpl_score',
            'obj1_token', 'obj1_token_id', 'agt2obj1_score', 'obj2_token', 'obj2_token_id', 'agt2obj2_score',
            'act4comet', 'edge4act'
        ]
        node_columns = ['head', 'edge', 'tail']
        score_columns = ['agt2act_score', 'node2act_score', 'node2node_score', 'total_score']
        columns = act_columns + node_columns + score_columns

        # get act_token_id
        all_score_df = all_score_df.loc[:, columns]
        if self.params['act_sampling_type'] == 'max':
            topi_score_df = all_score_df.loc[:, ['batch', 'total_score']].groupby('batch').idxmax()
            topi_score_df_ori = all_score_df.loc[topi_score_df.loc[:, 'total_score'], :]
            topi_score_df_ori.loc[:, 'prob'] = [1] * len(topi_score_df_ori)
            topi_score_df = topi_score_df_ori.sort_values(by=['batch']).reset_index(drop=True)
        else:
            topi_score_df = all_score_df.loc[:, ['batch', 'tmpl_token_id', 'total_score']]
            topi_score_df = topi_score_df.groupby(['batch', 'tmpl_token_id']).idxmax()
            topi_score_df = all_score_df.loc[topi_score_df.loc[:, 'total_score'], :].reset_index(drop=True)

            dist_n_sample, index_aggregate, prob_act = [], 0, []
            for b in range(self.batch_size):
                topi_score_dist = topi_score_df.loc[topi_score_df.loc[:, 'batch'] == b, 'total_score'].tolist()
                topi_score_dist = F.softmax(torch.tensor(topi_score_dist), dim=-1)
                topi_score_dist[topi_score_dist != topi_score_dist] = 0  # remove all nan value
                prob_act.extend((topi_score_dist + torch.tensor(1e-7)).tolist())
                topi_sample = (topi_score_dist + torch.tensor(1e-7)).multinomial(num_samples=1).squeeze(-1).tolist()
                dist_n_sample.append([topi_score_dist, topi_sample + index_aggregate])
                index_aggregate += topi_score_dist.size(-1)

            topi_score_df_ori = topi_score_df
            topi_score_df_ori.loc[:, 'prob'] = prob_act
            topi = [_i[1] for _i in dist_n_sample]
            topi_score_df = topi_score_df_ori.iloc[topi, :].sort_values(by=['batch']).reset_index(drop=True)

        topi_score = [int(_i) for _i in topi_score_df.loc[:, 'tmpl_token_id'].tolist()]
        columns = ['batch', 'tmpl_token_id', 'total_score']
        all_score_df = pd.merge(all_score_df, topi_score_df_ori.loc[:, columns+['prob']], how='outer', on=columns)
        all_score_df.loc[:, 'prob'] = all_score_df.loc[:, 'prob'].fillna(0)

        all_score_df.loc[:, 'tmpl_token'] = all_score_df.loc[:, 'tmpl_token'].fillna(0)
        assert len(all_score_df.loc[all_score_df.loc[:, 'tmpl_token'] == 0, :]) == 0

        index_aggregate, idx = 0, []
        for i, topi_score_i in enumerate(topi_score):
            idx_i = tmpl_token_id[index_aggregate:index_aggregate+tmpl_token_id_batch[i]]
            idx.append((idx_i == topi_score_i).nonzero(as_tuple=True)[0].item() + index_aggregate)
            index_aggregate += tmpl_token_id_batch[i]

        tmpl = (tmpl_token_id[idx, :], tmpl_dist[idx, :], [obj_num_in_tmpl[i] for i in idx])
        obj1 = (obj1_token_id[idx, :], obj1_dist[idx, :])
        obj2 = (obj2_token_id[idx, :], obj2_dist[idx, :])

        return tmpl, obj1, obj2, all_score_df
