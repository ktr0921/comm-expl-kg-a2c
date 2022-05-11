import os
import sys
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

from models.components import *

from models.comet import Comet
from models.cskg_construction import CSKGConstruction
from models.conditioning import COMeTExploration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pyhsical_entity = [
    'ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty', 'CapableOf', 'Desires', 'NotDesires'
]
event_centered = [
    'isAfter', 'HasSubEvent', 'IsBefore', 'HinderedBy', 'Causes', 'xReason', 'isFilledBy',
]
social_interaction = [
    'xEffect', 'xIntent', 'xNeed', 'xWant',
    'xAttr', 'xReact', 'oEffect', 'oReact', 'oWant',
]
# pyhsical_entity_guess = [  # this is not written in the paper, but it seems like from conceptnet.
#     # HOWEVER, DONT USE IT BECUZ I CANNOT FIND THIS IN ATOMIC2020 DATASET
#     "CausesDesire", "CreatedBy", "DefinedAs", "DesireOf", 'NotHasProperty', 'NotCapableOf',
#     "HasA", "HasFirstSubevent", "HasLastSubevent", "HasPainCharacter", "HasPainIntensity", "HasPrerequisite",
#     "InheritsFrom", "InstanceOf", "IsA", "LocatedNear",
#     "LocationOfAction", "MadeOf", "MotivatedByGoal", "NotHasA",
#     "NotIsA", "NotMadeOf", "PartOf", "ReceivesAction", "RelatedTo", "SymbolOf",
#     "UsedFor",
# ]
direction_tmpl = {
    'n': 'north', 's': 'south', 'w': 'west', 'e': 'east',
    'north': 'north', 'south': 'south', 'west': 'west', 'east': 'east',
    'nw': 'northwest', 'sw': 'southwest', 'ne': 'northeast', 'se': 'southeast',
    'northwest': 'northwest', 'southwest': 'southwest', 'northeast': 'northeast', 'southeast': 'southeast',
    'u': 'up', 'd': 'down', 'up': 'up', 'down': 'down',
    'front': 'front', 'back': 'back', 'right': 'right', 'left': 'left',
    'in': 'in', 'out': 'out',
}
direction_tmpl_tmp = []
for k, v in direction_tmpl.items():
    direction_tmpl_tmp += [k, v]
direction_tmpl_tmp = set(direction_tmpl_tmp)
atomic_2020_edge = pyhsical_entity + event_centered + social_interaction


class COMeTKGA2C(nn.Module):
    def __init__(
            self, params, vocab_id2tmpl, max_word_length, vocab_id2tkn, vocab_tkn2id, input_vocab_size, sp=None, gat=True,
    ):
        super().__init__()
        self.params = params

        self.vocab_id2tmpl = vocab_id2tmpl
        self.max_word_length = max_word_length
        self.vocab_id2tkn = vocab_id2tkn
        self.vocab_tkn2id = vocab_tkn2id
        self.input_vocab_size = input_vocab_size

        self.gat = gat

        # direction_tmpl
        self.direction_tmpl_idx = []
        for i, t in enumerate(vocab_id2tmpl):
            if any(t == d for d in direction_tmpl_tmp):
                self.direction_tmpl_idx.append(i)

        # Embeddings
        self.action_emb = nn.Embedding(len(vocab_id2tkn), params['embedding_size'])
        self.state_emb = nn.Embedding(input_vocab_size, params['embedding_size'])

        # observation & action layers
        self.action_drqa = ActionDrQA(
            input_vocab_size, params['embedding_size'], params['hidden_size'], params['batch_size'], params['recurrent']
        )
        self.tmpl_dec = DecoderRNN(  # [TEMPLATE DECODER]
            params['hidden_size'], len(vocab_id2tmpl)
        )
        self.tmpl_enc = EncoderLSTM(  # [TEMPLATE ENCODER]
            input_vocab_size, params['embedding_size'], int(params['hidden_size'] / 2), params['padding_idx'],
            params['dropout_ratio'], self.action_emb
        )
        self.obj_dec = DecoderRNN2(  # [OBJECT DECODER]
            params['hidden_size'], len(self.vocab_id2tkn.keys()), self.action_emb, params['graph_dropout']
        )

        # GAT
        self.state_gat = StateNetwork(
            params['gat_emb_size'], vocab_id2tkn, params['embedding_size'], params['hidden_size'],
            params['dropout_ratio'], params['tsv_file']
        )
        if not self.gat:
            self.state_fc = nn.Linear(params['hidden_size'] + 10, params['hidden_size'])
        else:
            self.state_fc = nn.Linear(params['hidden_size'] + params['hidden_size'] + 10, params['hidden_size'])

        # others
        self.softmax = nn.Softmax(dim=1)
        self.critic = nn.Linear(params['hidden_size'], 1)

        self.batch_size = params['batch_size']

        # comet
        self.do_comm_expl = params['do_comm_expl']
        if params['comet_path'] is not None:
            self.COMeT = Comet(params['comet_path'])
            self.COMeT.model.zero_grad()
            self.comet_prob = params['comet_prob']
        else:
            self.COMeT = None
            self.comet_prob = 0.0

        # prev
        self.root_node, self.root_node_person2you = [], dict()

        # cskg construction
        cskg_construction_args = {
            # comet
            'comet': self.COMeT,
            'cskg_size': params['cskg_size'],
            'num_generate': params['num_generate'],
            'store_comet_output': params['store_comet_output'],
            # node2node
            'vv_edge': params['vv_edge'],
            'vv_edge_sampling_n': params['vv_edge_sampling_n'],
            'vv_edge_sampling_type': params['vv_edge_sampling_type'],  # ['all', 'random', 'predict']
            # others
            'batch_size': self.batch_size,
            'sp': sp,
            'params': params
        }
        self.CSKGConstruction = CSKGConstruction(**cskg_construction_args)

        # conditioning
        conditioning_args = {
            # comet
            'comet': self.COMeT,
            'cskg_size': params['cskg_size'],
            'act_sampling_type': params['act_sampling_type'],  # ['max', 'softmax']
            # relations
            'va_edge': params['va_edge'],
            'va_edge_sampling_n': params['va_edge_sampling_n'],
            'va_edge_sampling_type': params['va_edge_sampling_type'],  # ['all', 'random', 'predict']
            'sp': sp,
            # agent
            'batch_size': self.batch_size,
            'vocab_id2tmpl': self.vocab_id2tmpl,
            'vocab_id2tkn': self.vocab_id2tkn,
            # gamma
            'vv_prev_score_gamma': params['vv_prev_score_gamma'],
            'vv_score_gamma': params['vv_score_gamma'],
            'va_score_gamma': params['va_score_gamma'],
            'tmpl_score_gamma': params['tmpl_score_gamma'],
            'aa_score_gamma': params['aa_score_gamma'],
            # param
            'params': params
        }
        self.conditioning = COMeTExploration(**conditioning_args)

        # actions
        self.include_adm_act = params['include_adm_act']
        self.tmpl_max_number = params['tmpl_max_number']
        self.tmpl_min_prob   = (1 / len(vocab_id2tmpl)) * params['tmpl_min_prob']
        if len(self.vocab_id2tmpl) < self.tmpl_max_number:
            print(f'===\n{self.tmpl_max_number} is changed to {len(self.vocab_id2tmpl)}\n===')
            self.tmpl_max_number = len(self.vocab_id2tmpl)

        # entropy
        self.do_entropy_threshold = params['do_entropy_threshold']
        self.entropy_list_size    = params['entropy_list_size']
        self.entropy_list         = torch.zeros(self.batch_size).unsqueeze(-1).to(device)

    def get_action_rep(self, action):
        '''
        obj_num_in_tmpl: the number of objects required for the template
        '''
        action = str(action)
        obj_num_in_tmpl = action.count('OBJ')
        action = action.replace('OBJ', '')
        action_desc_num = 20 * [0]

        for i, token in enumerate(action.split()[:20]):
            short_tok = token[:self.max_word_length]
            action_desc_num[i] = self.vocab_tkn2id[short_tok] if short_tok in self.vocab_tkn2id else 0

        return action_desc_num, obj_num_in_tmpl

    def get_tmpl(self, state_emb, h_t, adm_tmpl, test=False):
        # get tmpl_distribution
        tmpl_distribution, tmpl_hidden = self.tmpl_dec(state_emb, h_t)
        softmax_tmpl_distribution = self.softmax(tmpl_distribution)

        if (self.do_comm_expl is True) and (test is False):
            # get prob
            # if self.do_entropy_threshold, condition it based on entropy
            if self.do_entropy_threshold:
                # get tmpl_entropy & get prob
                tmpl_entropy, prob = [], []
                for i in range(self.batch_size):
                    # tmpl_entropy_i => tmpl_entropy
                    softmax_tmpl_distribution_i = softmax_tmpl_distribution[i].detach().clone() + torch.tensor(1e-7)
                    tmpl_entropy_i = -(softmax_tmpl_distribution_i * softmax_tmpl_distribution_i.log()).mean().to(device)
                    tmpl_entropy.append(tmpl_entropy_i)

                    # prob_i => prob
                    # if size of entropy_list >= entropy_list_size, prob_i=entropy-prob
                    if self.entropy_list.size(-1) >= self.entropy_list_size:
                        # if self.entropy_list is not full, get scale of tmpl_entropy_i as prob, else 0 (run COMeT)
                        ordered_entropy = torch.cat([self.entropy_list[i], tmpl_entropy_i.unsqueeze(-1)], dim=-1)

                        tmpl_entropy_i_order = (ordered_entropy.sort(dim=-1)[0] == tmpl_entropy_i.squeeze(-1).item()) \
                            .nonzero(as_tuple=True)[0]
                        if self.params['test'] == 'entropy_threshold':
                            print(f'=== {i}')
                            print(ordered_entropy.sort(dim=-1)[0])
                            print(ordered_entropy.sort(dim=-1)[0] == tmpl_entropy_i.squeeze(-1).item())
                        prob_i = 1 - (tmpl_entropy_i_order.to(torch.float64).mean().item() + 1) / len(ordered_entropy)
                    # else, prob_i=-1 (Apply)
                    else:
                        prob_i = 2

                    prob.append(prob_i)
                # append tmpl_entropy => entropy_list_tmp
                entropy_list_tmp = [self.entropy_list, torch.stack(tmpl_entropy).to(device).unsqueeze(-1)]
                self.entropy_list = torch.cat(entropy_list_tmp, dim=-1)[:, -self.entropy_list_size:].to(device)
            # else, randomly select
            else:
                prob = [random.random()] * self.batch_size

            # based on condition (prob <= self.comet_prob), sample or select topk of tmpl_token_id
            condition, tmpl_token_id = [prob_i <= self.comet_prob for prob_i in prob], []
            for i in range(softmax_tmpl_distribution.size(0)):
                adm_tmpl_i = set(adm_tmpl[i])
                softmax_tmpl_distribution_i = softmax_tmpl_distribution[i].detach().clone() + torch.tensor(1e-7)
                if condition[i] is False:
                    tmpl_token_id_i = softmax_tmpl_distribution_i.multinomial(num_samples=1)
                elif self.tmpl_min_prob > 0:
                    # if too many action is larger than tmpl_min_prob, choose top k of tmpl_max_number
                    if (softmax_tmpl_distribution_i >= self.tmpl_min_prob).sum(-1) > self.tmpl_max_number:
                        tmpl_token_id_i = softmax_tmpl_distribution_i.topk(self.tmpl_max_number)[1]
                    # elif no action is larger than tmpl_min_prob (this only occurs when self.ignore_direction
                    # since if not, at least one action will be very high, but it may never occur even since entropy might
                    # be too small if distribution is skewed this much)
                    elif (softmax_tmpl_distribution_i >= self.tmpl_min_prob).sum(-1) == 0:
                        tmpl_token_id_i = (softmax_tmpl_distribution[i] + torch.tensor(1e-7)).multinomial(num_samples=1)
                    # else, choose all the actions that are larger than tmpl_min_prob
                    else:
                        tmpl_token_id_i = (softmax_tmpl_distribution_i >= self.tmpl_min_prob).nonzero(as_tuple=True)[0]

                    if self.include_adm_act:
                        tmpl_token_id_i = list(set(tmpl_token_id_i.tolist()) | adm_tmpl_i)
                        tmpl_token_id_i = torch.tensor(tmpl_token_id_i).to(device).to(torch.long)
                else:
                    tmpl_token_id_i = softmax_tmpl_distribution_i.topk(self.tmpl_max_number)[1]
                tmpl_token_id.append(tmpl_token_id_i)

            if self.params['test'] == 'entropy_threshold':
                print(f'tmpl_token_id:')
                for t in tmpl_token_id:
                    print(t)

        else:
            tmpl_token_id = [
                (softmax_tmpl_distribution[i].detach().clone() + torch.tensor(1e-7)).multinomial(num_samples=1)
                for i in range(softmax_tmpl_distribution.size(0))
            ]
            prob = [2] * self.batch_size
            condition = [False] * self.batch_size

        # For 'TMPL', self.vocab_id2tmpl[-] swaps idx to token {159: 'break OBJ with OBJ'}.
        # This can be represented as vocab, 'break': 47, 'OBJ': --, 'with': 78, 'OBJ': -- using self.vocab_tkn2id[-]
        # tmpl_vocab_id: Store 'TMPL' as vocab; ['break OBJ with OBJ'] => [47, 78, 0, 0, 0, ...]; len()=20
        # ['break OBJ with OBJ', 'take OBJ'] => [[47, 78, 0, ...], [23, 0, 0, ...]]; len()=max_word_length=20
        # obj_num_in_tmpl: Store the number of 'OBJ' in 'TEMPL'; 'break OBJ with OBJ' => 2
        # ['break OBJ with OBJ', 'take OBJ'] => [2, 1]
        tmpl_vocab_id, obj_num_in_tmpl = [], []
        tmpl_token_id_batch = [_ts.size(0) for _ts in tmpl_token_id]

        for i in range(self.batch_size):  # self.batch_size = number_of_rl_environments
            for j in range(tmpl_token_id[i].size(0)):
                tmpl_str = self.vocab_id2tmpl[tmpl_token_id[i][j].detach().item()]
                tmpl_vocab_id_i, obj_num_in_tmpl_i = self.get_action_rep(tmpl_str)
                tmpl_vocab_id.append(tmpl_vocab_id_i)
                obj_num_in_tmpl.append(obj_num_in_tmpl_i)

        tmpl_vocab_id_ts = torch.tensor(tmpl_vocab_id).to(device).clone()
        _, tmpl2obj_init_h, _ = self.tmpl_enc(tmpl_vocab_id_ts)

        return tmpl_distribution, tmpl_hidden, tmpl_token_id, tmpl_token_id_batch, obj_num_in_tmpl, tmpl2obj_init_h, \
               prob, condition

    def get_objk(self, objk_dec_in, objk_hidden, tmpl2obj_init_h, graphs):
        objk_distribution, _, objk_hidden = self.obj_dec(objk_dec_in, objk_hidden, tmpl2obj_init_h, graphs)
        softmax_objk_distribution, objk_token_id = torch.zeros([graphs.size(0), objk_distribution.size(-1)]).to(device), []

        agt_obj_n = 1  # agt_obj_n = self.agt_obj_n if prob <= self.comet_prob else 1
        for i in range(graphs.size(0)):
            agt_obj_n = agt_obj_n if agt_obj_n < len(graphs[i]) else len(graphs[i])
            softmax_graph_objk_distribution_i = F.softmax(objk_distribution[i][graphs[i]], dim=0)
            graph_objk_token_id_i = (softmax_graph_objk_distribution_i + torch.tensor(1e-7)).multinomial(agt_obj_n)

            graph_list = graphs[i].nonzero().cpu().numpy().flatten().tolist()  # assert
            assert len(graph_list) == softmax_graph_objk_distribution_i.numel()

            softmax_objk_distribution[i, graphs[i]] += softmax_graph_objk_distribution_i.detach().clone()
            objk_token_id.append([graph_list[graph_objk_token_id_ij] for graph_objk_token_id_ij in graph_objk_token_id_i])

        objk_token_id = torch.LongTensor(objk_token_id).to(device)

        return objk_distribution, objk_hidden, objk_token_id, softmax_objk_distribution

    def forward(self, obs, scores, graph_representation, graph, adm_tmpl, test=False):
        '''
        :param obs: The encoded ids for the textual observations (shape 4x300):
        The 4 components of an observation are: look - ob_l, inventory - ob_i, response - ob_r, and prev_action.
        :type obs: ndarray

        number_of_rl_environments=16
        hidden_size=100
        max_word_length=20
        num_directions=1 (NOT Bidirectional)
        '''

        # [MIDDLE] encode observation (o_{desc,t}, o_{game,t}, o_{inv,t}, a_{t-1}) => vector o_{t} and h_{t} (recurrent)
        o_t, h_t = self.action_drqa.forward(obs)

        # [RIGHT] Binary Score Encoding
        src_t = []
        for scr in scores:
            # fist bit encodes +/-
            if scr >= 0:
                cur_st = [0]
            else:
                cur_st = [1]
            # if scr = scores[i] = 3, cur_st = src_t[i] = 0000000011 AND scores[i] = -5, src_t[i] = 0000000101
            cur_st.extend([int(c) for c in '{0:09b}'.format(abs(scr))])
            src_t.append(cur_st)
        src_t = torch.FloatTensor(src_t).to(device)

        # [LEFT] GAT to state embedding => pass state_emb to self.state_fc()
        if not self.gat:
            state_emb = torch.cat((o_t, src_t), dim=1)       # concat o_t & src_t
        else:
            g_t = self.state_gat.forward(graph_representation)
            state_emb = torch.cat((g_t, o_t, src_t), dim=1)  # concat g_t & o_t & src_t
        state_emb = F.relu(self.state_fc(state_emb))

        # === [CRITIC] ===
        state_value = self.critic(state_emb.clone())  # might need .detach()

        # === [TMPL] ===
        tmpl_distribution, tmpl_hidden, tmpl_token_id, tmpl_token_id_batch, obj_num_in_tmpl, tmpl2obj_init_h, \
            prob, condition = self.get_tmpl(state_emb, h_t, adm_tmpl, test)

        tmpl_hidden_, graph_, tmpl_distribution_ = [], [], []
        for i in range(tmpl_hidden.size(1)):
            tmpl_hidden_i = tmpl_hidden[:, i, :].unsqueeze(1).expand(-1, tmpl_token_id_batch[i], -1)
            graph_i = graph[i, :].unsqueeze(0).expand(tmpl_token_id_batch[i], -1)
            tmpl_distribution_i = tmpl_distribution[i, :].unsqueeze(0).expand(tmpl_token_id_batch[i], -1)
            tmpl_hidden_.append(tmpl_hidden_i)
            graph_.append(graph_i)
            tmpl_distribution_.append(tmpl_distribution_i)
        tmpl_hidden = torch.cat(tmpl_hidden_, dim=1)
        graph = torch.cat(graph_, dim=0)
        tmpl_distribution = torch.cat(tmpl_distribution_, dim=0)

        # === [OBJ1] ===
        obj1_dec_in = torch.tensor([self.vocab_tkn2id['<s>']] * tmpl2obj_init_h.size(0)).to(device)
        obj1_hidden = tmpl_hidden
        obj1_distribution, obj1_hidden, obj1_token_id, softmax_obj1_distribution = \
            self.get_objk(obj1_dec_in, obj1_hidden, tmpl2obj_init_h, graph)

        # === [OBJ2] ===
        obj2_dec_in = obj1_token_id.squeeze(-1).detach()
        obj2_hidden = obj1_hidden
        obj2_distribution, obj2_hidden, obj2_token_id, softmax_obj2_distribution = \
            self.get_objk(obj2_dec_in, obj2_hidden, tmpl2obj_init_h, graph)

        tmpl_token_id = torch.cat(tmpl_token_id, dim=0).unsqueeze(-1).to(device)
        tmpl = (tmpl_token_id, tmpl_distribution, obj_num_in_tmpl)
        obj1 = (obj1_token_id, obj1_distribution)
        obj2 = (obj2_token_id, obj2_distribution)

        if (self.do_comm_expl is False) or (self.COMeT is None) or (sum(condition) == 0):
            cskg_df, all_score_df = None, None
        else:
            cskg_df = self.CSKGConstruction.get_cskg(self.root_node, self.root_node_person2you)
            tmpl, obj1, obj2, all_score_df = self.conditioning.comet_conditioning(
                tmpl, obj1, obj2, graph, tmpl_token_id_batch, cskg_df,
                self.CSKGConstruction.Node2NodeEdgeSampling, self.root_node_person2you,
            )

        # tmpl_distribution: (number_of_rl_environments, tmpl_space)
        # obj_distribution: (n_of_obj_for_tmpl, number_of_rl_environments, obj_space); this is for only 1 templ
        # obj_token_id: (max_decode_steps=2, number_of_rl_environments)
        # tmpl_token_id.size() = (number_of_rl_environments, self.tmpl_max_number=3)
        # state_value: output from [CRITIC]
        # obj_num_in_tmpl: Store the number of 'OBJ' in 'TEMPL'; 'break OBJ with OBJ': 2
        return state_value, tmpl, obj1, obj2, prob, cskg_df, all_score_df

    def clone_hidden(self):
        self.action_drqa.clone_hidden()

    def restore_hidden(self):
        self.action_drqa.restore_hidden()

    def reset_hidden(self, done_mask_tt):
        self.action_drqa.reset_hidden(done_mask_tt)
