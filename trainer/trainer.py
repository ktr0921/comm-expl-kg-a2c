import os
from os.path import basename, splitext
import numpy as np
import time
import re
import sentencepiece as spm
import string
import random
from datetime import datetime
from collections import Counter
# from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from jericho.util import unabbreviate, clean
import jericho.defines

from models.models import COMeTKGA2C
from utils.representations import StateAction
from utils.env import *
from utils.vec_env import *
from utils import logger
from models.comet import Comet
# from run import cuda_number

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

walkthrough = [  # kelvin test
    'N', 'N', 'U', 'Get egg', 'D', 'S', 'E', 'Open window', 'W', 'Open sack', 'Get garlic', 'W', 'Get lamp', 'E', 'U',
    'Light lamp', 'Get rope', 'Get knife', 'D', 'Douse lamp', 'W', 'Get sword', 'Move rug', 'Open trapdoor', 'D',
    'Light lamp', 'N', 'Kill troll with sword', 'drop egg', 'E', 'E', 'Se', 'E', 'Tie rope to railing', 'D',
    'Douse lamp', 'Get torch', 'D', 'S', 'Drop sword', 'Get candles', 'Douse candles', 'Get book', 'N', 'Get bell',
    'E', 'Open coffin', 'Get sceptre', 'W', 'S', 'Pray', 'E', 'S', 'E', 'W', 'W', 'Read book', 'Drop all', 'get torch',
    'get lamp', 'Open trapdoor', 'D', 'N', 'E', 'E', 'Se', 'E', 'D', 'D', 'D', 'Get coffin', 'U', 'S', 'Pray', 'E',
    'S', 'E', 'W', 'W', 'Open case', 'Put coffin in case', 'Get book', 'Get bell', 'Get candles', 'D', 'S', 'E',
    'Get painting', 'W', 'N', 'N', 'E', 'E', 'E', 'Echo', 'Get bar', 'U', 'E', 'N', 'Drop painting', 'Get matchbook',
    'S', 'S', 'D', 'W', 'S', 'S', 'E', 'D', 'Ring bell', 'Get candles', 'Light match', 'Light candles with match',
    'Read prayer', 'Drop matchbook', 'Drop candles', 'Drop book', 'S', 'get skull', 'N', 'U', 'N', 'N', 'N', 'W', 'W',
    'S', 'U', 'Put skull in case', 'Put bar in case', 'D', 'N', 'W', 'W', 'W', 'U', 'Get knife', 'get bag', 'Sw', 'E',
    'S', 'Se', 'Odysseus', 'E', 'E', 'Put bag in case', 'Drop rusty knife', 'D', 'N', 'E', 'E', 'N', 'Ne', 'E', 'D',
    'Get pile', 'U', 'N', 'N', 'Get screwdriver', 'Get wrench', 'Press red button', 'Press yellow button', 'S', 'S',
    'Turn bolt with wrench', 'Drop wrench', 'W', 'W', 'E', 'Sw', 'S', 'S', 'W', 'W', 'S', 'U', 'Drop pile',
    'Drop screwdriver', 'D', 'N', 'E', 'E', 'S', 'S', 'Touch mirror', 'E', 'D', 'Get trident', 'S', 'Get pump', 'S',
    'S', 'Sw', 'S', 'S', 'W', 'W', 'S', 'U', 'Put trident in case', 'Get sceptre', 'Get pile', 'D', 'N', 'E', 'E',
    'E', 'E', 'E', 'Drop pile', 'Inflate pile', 'Drop pump', 'Get label', 'Read label', 'Drop label',
    'Throw sceptre in boat', 'Enter boat', 'Launch', 'Get sceptre', 'Wait', 'Get buoy', 'Wait', 'Land',
    'Get out of boat', 'N', 'Open buoy', 'Get emerald', 'Drop buoy', 'Get shovel', 'Ne', 'Dig sand', 'Dig sand',
    'Dig sand', 'Dig sand', 'Drop shovel', 'Get scarab', 'Sw', 'S', 'S', 'Wave sceptre', 'W', 'W', 'Get pot', 'Sw',
    'U', 'U', 'Nw', 'W', 'W', 'W', 'Put sceptre in case', 'Put pot in case', 'Put emerald in case',
    'Put scarab in case', 'Get rusty knife', 'Get nasty knife', 'W', 'W', 'U', 'Kill thief with nasty knife',
    'kill thief with nasty knife', 'kill thief with nasty knife', 'attack thief with nasty knife', 'Get painting',
    'Get egg', 'Drop rusty knife', 'Drop nasty knife', 'Get chalice', 'D', 'E', 'E', 'Put painting in case',
    'Get canary', 'Put chalice in case', 'Put egg in case', 'E', 'E', 'N', 'N', 'E', 'Wind canary', 'Get bauble', 'W',
    'S', 'E', 'W', 'W', 'Put canary in case', 'Put bauble in case', 'Get screwdriver', 'Get garlic', 'D', 'N', 'E',
    'E', 'S', 'S', 'Touch mirror', 'N', 'W', 'N', 'W', 'N', 'E', 'Put torch in basket', 'Put screwdriver in basket',
    'Light lamp', 'N', 'D', 'E', 'Ne', 'Se', 'Sw', 'D', 'D', 'S', 'Get coal', 'N', 'U', 'U', 'N', 'E', 'S', 'N', 'U',
    'S', 'Put coal in basket', 'Lower basket', 'N', 'D', 'E', 'Ne', 'Se', 'Sw', 'D', 'D', 'W', 'Drop all', 'W',
    'Get all from basket', 'S', 'Open lid', 'Put coal in machine', 'Close lid', 'Turn switch with screwdriver',
    'Open lid', 'Get diamond', 'N', 'Put diamond in basket', 'Put torch in basket', 'Put screwdriver in basket', 'E',
    'Get lamp', 'Get garlic', 'E', 'U', 'U', 'N', 'E', 'S', 'N', 'Get bracelet', 'U', 'S', 'Raise basket',
    'Get all from basket', 'W', 'Get figurine', 'S', 'E', 'S', 'D', 'Drop garlic', 'U', 'Put diamond in case',
    'Put torch in case', 'Put bracelet in case', 'Put figurine in case', 'W', 'W', 'U', 'take trunk', 'D', 'E',
    'E', 'PUT trunk in case', 'Look', 'Get map', 'Examine map', 'E', 'E', 'N', 'W', 'Sw', 'W'
]


def configure_logger(log_dir):
    print("save log at: {}".format(log_dir))
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log


class KGA2CTrainer(object):
    def __init__(self, params):
        print(f"===== initiating under {device} =====")
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            print(f'=== current_device: {current_device}')
            print(f'=== device_count: {torch.cuda.device_count()}')
            print(f'=== get_device_name: {torch.cuda.get_device_name(current_device)}')
            memory_cached = torch.cuda.memory_cached(current_device)
            memory_allocated = torch.cuda.memory_allocated(current_device)
            memory_free = memory_cached - memory_allocated
            print(f'=== memory_cached: {memory_cached}')
            print(f'=== memory_allocated: {memory_allocated}')
            print(f'=== memory_cached - memory_allocated = memory_free: {memory_free}')

        self.params = params
        torch.set_printoptions(profile="full")

        print("step 0 seed")
        torch.manual_seed(params['seed'])
        np.random.seed(params['seed'])
        random.seed(params['seed'])

        print("step 1 configure logger")
        self.output_dir = self.params['output_dir']
        os.makedirs(self.output_dir)
        os.makedirs(os.path.join(self.output_dir, 'cskg'))
        configure_logger(self.output_dir)

        print("step 2 build sp")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(params['spm_file'])

        print("step 3 build kga2c env")
        kg_env = KGA2CEnv(
            params['rom_file_path'], params['seed'], self.sp, params['tsv_file'], step_limit=params['reset_steps'],
            stuck_steps=params['stuck_steps'], gat=params['gat'],
            redis_port=params['redis_port'], openie_port=params['openie_port'],
        )

        print("step 4 build env")
        self.vec_env = VecEnv(
            params['batch_size'], kg_env,
            params['openie_path'], params['openie_port'],
            params['redis_path'], params['redis_config_path'], params['redis_port'],
        )
        # openie_path, openie_port, redis_path, redis_config_path, redis_port
        env = FrotzEnv(params['rom_file_path'])
        self.vocab_act, self.vocab_act_rev = load_vocab(env)

        print("step 5 build templace generator")
        self.binding = load_bindings(params['rom_file_path'])
        self.template_generator = TemplateActionGenerator(self.binding)

        print("step 6 build kga2c model")
        # init kga2c model
        self.max_word_length = self.binding['max_word_length']
        self.model = COMeTKGA2C(
            params, self.template_generator.templates, self.max_word_length, self.vocab_act, self.vocab_act_rev,
            len(self.sp), self.sp, gat=self.params['gat'],
        ).to(device)

        # load pretrained or train from scratch
        if params['preload_weights']:
            print("load pretrained")

            # self.model = torch.load(self.params['preload_weights'], map_location=device)['model']
            number = self.params['preload_weights'].split('-')[-1].split('.')[0]
            number = f'-{number}' if number.isdigit() else ''
            parent = self.params['preload_weights'].rsplit('/', 1)[0]
            self.model.load_state_dict(torch.load(self.params['preload_weights'], map_location=device))
            entropy_list = os.path.join(parent, f'saved_entropy_data{number}.pt')
            self.model.entropy_list = torch.load(entropy_list, map_location=device)

            print(f'entropy_list: {self.model.entropy_list}')
        else:
            print("train from scratch")

        print("step 7 set training parameters")
        # others
        self.batch_size = params['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), lr=params['lr'])
        self.loss_fn1 = nn.BCELoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss()
        self.loss_fn3 = nn.MSELoss()

        print("step 8 parameters")
        log('Parameters {}'.format(params))

        print("===== /initiating =====")

    def generate_targets(self, admissible, objs):
        '''
        Generates ground-truth targets for admissible actions.

        :param admissible: List-of-lists of admissible actions. Batch_size x Admissible
        :param objs: List-of-lists of interactive objects. Batch_size x Objs
        :returns: template targets and object target tensors
        '''
        tmpl_target = []
        obj_targets = []
        for adm in admissible:
            obj_t = set()
            cur_t = [0] * len(self.template_generator.templates)
            for a in adm:
                cur_t[a.template_id] = 1
                obj_t.update(a.obj_ids)
            tmpl_target.append(cur_t)
            obj_targets.append(list(obj_t))
        tmpl_target_tt = torch.FloatTensor(tmpl_target).to(device)

        # Note: Adjusted to use the objects in the admissible actions only
        object_mask_target = []
        for objl in obj_targets:  # in objs
            cur_objt = [0] * len(self.vocab_act)
            for o in objl:
                cur_objt[o] = 1
            object_mask_target.append([[cur_objt], [cur_objt]])
        obj_target_tt = torch.FloatTensor(object_mask_target).squeeze().to(device)
        return tmpl_target_tt, obj_target_tt

    def generate_graph_mask(self, graph_infos):
        assert len(graph_infos) == self.batch_size
        mask_all = []
        for graph_info in graph_infos:
            mask = [0] * len(self.vocab_act.keys())
            if self.params['masking'] == 'kg':
                # Uses the knowledge graph as the mask.
                graph_state = graph_info.graph_state
                ents = set()
                for u, v in graph_state.edges:
                    ents.add(u)
                    ents.add(v)
                for ent in ents:
                    for ent_word in ent.split():
                        if ent_word[:self.max_word_length] in self.vocab_act_rev:
                            idx = self.vocab_act_rev[ent_word[:self.max_word_length]]
                            mask[idx] = 1
            elif self.params['masking'] == 'interactive':
                # Uses interactive objects grount truth as the mask.
                for o in graph_info.objs:
                    o = o[:self.max_word_length]
                    if o in self.vocab_act_rev.keys() and o != '':
                        mask[self.vocab_act_rev[o]] = 1
            elif self.params['masking'] == 'none':
                # No mask at all.
                mask = [1] * len(self.vocab_act.keys())
            else:
                assert False, 'Unrecognized masking {}'.format(self.params['masking'])
            mask_all.append(mask)
        return torch.BoolTensor(mask_all).to(device).detach()

    def discount_reward(self, transitions, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(transitions))):
            state_value, rewards, done_masks, _, _, _, _, _, _ = transitions[t]
            R = rewards + self.params['gamma'] * R * done_masks
            adv = R - state_value
            returns.append(R)
            advantages.append(adv)
        return returns[::-1], advantages[::-1]

    def get_root_node(self, obs, obs_look, prev_obs_look, chosen_act, act_inc):
        # empty
        self.model.root_node, self.model.root_node_person2you = [], dict()
        self.model.root_node = []
        self.model.root_node_person2you = dict()
        # direction tmpl
        dir_tmpl = {
            'n': 'north', 's': 'south', 'w': 'west', 'e': 'east',
            'north': 'north', 'south': 'south', 'west': 'west', 'east': 'east',
            'nw': 'northwest', 'sw': 'southwest', 'ne': 'northeast', 'se': 'southeast',
            'northwest': 'northwest', 'southwest': 'southwest', 'northeast': 'northeast', 'southeast': 'southeast',
            'u': 'up', 'd': 'down', 'up': 'up', 'down': 'down',
            'front': 'front', 'back': 'back', 'right': 'right', 'left': 'left',
            'in': 'in', 'out': 'out',
        }

        # for every batch, create root node
        for obs_i, obs_look_i, prev_obs_look_i, chosen_act_i, act_inc_i in zip(obs, obs_look, prev_obs_look, chosen_act, act_inc):
            # change abbreviated direction tmpl to readable direction tmpl
            # 'n', 's', ... & 'nw', 'sw', ... & 'u', 'd' => 'north', 'south', ... & 'northwest', 'southwest' & 'up', 'down'
            chosen_act_i = dir_tmpl[chosen_act_i.lower()] if chosen_act_i.lower() in dir_tmpl.keys() \
                else chosen_act_i.lower()
            chosen_act_i = 'go ' + chosen_act_i if chosen_act_i in dir_tmpl.values() else chosen_act_i

            if type(obs_look_i) == list:
                obs_look_i = str(obs_look_i[0]) if obs_look_i else ''
            if type(prev_obs_look_i) == list:
                prev_obs_look_i = str(prev_obs_look_i[0]) if prev_obs_look_i else ''
            obs_i, obs_look_i, prev_obs_look_i = obs_i.strip(), obs_look_i.strip(), prev_obs_look_i.strip()

            # remove score functions
            regex = [
                r'\[Your score has just gone up by [\w\-]+ point[s]?.\]',
                r'\[Your score has just gone down by [\w\-]+ point[s]?.\]',
                r'\[Your score just went up by [\d]+ point[s]?. The total is now [\d]+ out of 100.\]',
                r'\[Your score just went down by [\d]+ point[s]?. The total is now [\d]+ out of 100.\]',
                r'\[My score has just gone up by [\w\-]+ point[s]?.\]',
                r'\[My score has just gone down by [\w\-]+ point[s]?.\]',
                r'\[Grunk score go up [\w\-]+.\]',
                r'\[Grunk score go down [\w\-]+.\]',
                r"\[Nikolai's sanitation rating has dropped [\w\-]+ point[s]?.\]",
                r'\[[\d]+\]',
            ]
            for regex_i in regex:
                obs_i = re.sub(regex_i, '', obs_i)
            obs_i = re.sub(r'\[.*?\]', '', obs_i)
            # # if nswe tmpl or look, previous action + game feedback
            # # otherwise, room description + previous action + game feedback
            # if chosen_act_i in dir_tmpl.values():
            #     if obs_look_i in obs_i:
            #         root_node = f'{obs_i}'
            #     elif obs_i in obs_look_i:
            #         root_node = f'{obs_look_i}'
            #     else:
            #         print('===== ===== ===== ===== =====')
            #         print('obs_i and obs_look_i do not overlap, so use longer')
            #         print('obs_i:', obs_i)
            #         print('obs_look_i:', obs_look_i)
            #         print('===== ===== ===== ===== =====')
            #         obs_use = obs_i if len(obs_i) > len(obs_look_i) else obs_look_i
            #         root_node = f'{obs_use}'
            # el
            if act_inc_i is False:
                root_node = f'{obs_look_i}' if obs_look_i else f'{prev_obs_look_i}'
                # if len(obs_look_i) == 0:
                #     print('obs_look_i')
                #     print(obs_look_i)
                #     print('prev_obs_look_i')
                #     print(prev_obs_look_i)
            elif len(obs_i.strip().split(' ')) <= 20:  # can change it to 5
                root_node = f'{prev_obs_look_i}. You {chosen_act_i}. {obs_i}'
            else:
                root_node = f'You {chosen_act_i}. {obs_i}'
            root_node = root_node.strip('. ').strip()
            root_node = f'{root_node}.' if (root_node[-1].isalpha()) or (root_node[-1].isdigit()) else root_node
            root_node = re.sub(r'[ ]*\.[\. ]*', '. ', re.sub(r'\n+', '.', root_node))
            root_node = ' '.join(root_node.split())

            # modification from get_obs_rep & get_visible_state_rep_drqa in representations.StateAction
            remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']
            for rm in remove:
                root_node = root_node.replace(rm, '')

            # create root node of personx version
            root_node_personx = root_node
            node_change = {
                'You are': 'PersonX is', 'you are': 'PersonX is',
                'Are you': 'Is PersonX', 'are you': 'is PersonX',
                'You do': 'PersonX does', 'you do': 'PersonX does',
                'Do you': 'Does PersonX', 'do you': 'does PersonX',
                'You': 'PersonX', 'you': 'PersonX',
                'Your': "PersonX's", 'your': "PersonX's",
            }
            for k, v in node_change.items():
                root_node_personx = root_node_personx.replace(k, v)

            root_node = (root_node[0].upper() + root_node[1:]).strip()
            root_node_personx = (root_node_personx[0].upper() + root_node_personx[1:]).strip()

            self.model.root_node.append(root_node_personx)                  # with batch, create root node
            self.model.root_node_person2you[root_node_personx] = root_node  # transform back to original

    def train(self, max_steps):
        print("=== === === start training!!! === === ===")
        start = time.time()
        transitions = []
        # obs[i] is textual description
        # infos[i] is any other info, {'moves':0, 'score':0, 'valid':False, 'steps':0} &
        # graph_infos[i] contains all of objs, ob_rep, act_rep, graph_state, graph_state_rep, admissible_actions, admissible_actions_rep
        obs, infos, graph_infos = self.vec_env.reset()  # "class KGA2CEnv" in env.py
        chosen_act, tmpl_token_id, prev_obs_look, init_obs = \
            tuple(['look'] * self.batch_size), torch.tensor([-1] * self.batch_size), obs, obs[0]
        dones = [False] * self.batch_size
        for step in range(1, max_steps + 1):
            # model()
            # obs_reps[:, (0, 1, 2, 3), :] = (room description, inventory, game feedback, previous action)
            obs_reps = np.array([g.ob_rep for g in graph_infos])         # obs_rep
            scores = [info['score'] for info in infos]                   # score
            graph_state_reps = [g.graph_state_rep for g in graph_infos]  # graph_state_reps
            graph_mask = self.generate_graph_mask(graph_infos)           # masking on kg/interactive/none options
            adm_tmpl = [[a.template_id for a in g.admissible_actions] for g in graph_infos]

            if self.params['do_comm_expl'] is True:
                # get root node
                if self.model.conditioning.COMeT:
                    obs_look = [graph_infos_i.observation_look for graph_infos_i in graph_infos]
                    chosen_act = ['look' if _j.strip() == init_obs.strip() else _i for _i, _j in zip(chosen_act, obs)]
                    act_inc = [True if tmpl_token_id_i in adm_tmpl_i else False
                               for tmpl_token_id_i, adm_tmpl_i in zip(tmpl_token_id, adm_tmpl)]
                    self.get_root_node(obs, obs_look, prev_obs_look, chosen_act, act_inc)
                    prev_obs_look = obs_look

            # model
            state_value, tmpl, obj1, obj2, prob, cskg_df, all_score_df = self.model(
                obs_reps, scores, graph_state_reps, graph_mask, adm_tmpl
            )
            tmpl_token_id, tmpl_distribution, obj_num_in_tmpl = tmpl
            obj1_token_id, obj1_distribution = obj1
            obj2_token_id, obj2_distribution = obj2

            # Generate the ground truth and object mask
            admissible = [g.admissible_actions for g in graph_infos]
            objs = [g.objs for g in graph_infos]
            tmpl_gt, obj_mask_gt = self.generate_targets(admissible, objs)

            # Log template predictions (tmpl_gt) and ground truth in graph (obj_mask_gt)
            tmpl_distribution_0 = F.softmax(tmpl_distribution[0].detach().clone(), dim=-1) + torch.tensor(1e-7)
            tmpl_entropy = -(tmpl_distribution_0 * tmpl_distribution_0.log()).mean().item()
            tmpl_prob_topk, tmpl_id_topk = tmpl_distribution_0.topk(20)
            tmpl_topk = [self.template_generator.templates[t] for t in tmpl_id_topk.tolist()]
            # [TEMPL] [PROBABILITY], [TEMPL] [PROBABILITY], ...
            tmpl_pred_log = ', '.join([f'{_a} {round(_b, 3)}' for _a, _b in zip(tmpl_topk, tmpl_prob_topk.tolist())])
            tmpl_gt_log = ', '.join([self.template_generator.templates[i] for i in tmpl_gt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()])

            # Log object predictions (obj_distribution) and ground truth in graph (obj_mask_gt)
            objk_prob_topk, objk_id_topk = F.softmax(torch.stack([obj1_distribution[0], obj2_distribution[0]]), dim=-1).topk(5)
            obj1_id_topk, obj2_id_topk = [[self.vocab_act[_ij] for _ij in _i] for _i in objk_id_topk.tolist()]
            # [OBJ] [PROBABILITY], [OBJ] [PROBABILITY], ...
            obj1_pred_log = ', '.join(['{} {:.3f}'.format(o, o_prob) for o, o_prob in zip(obj1_id_topk, objk_prob_topk[0].tolist())])
            obj2_pred_log = ', '.join(['{} {:.3f}'.format(o, o_prob) for o, o_prob in zip(obj2_id_topk, objk_prob_topk[1].tolist())])
            objk_gt_log = ', '.join([self.vocab_act[i] for i in obj_mask_gt[0, 0].nonzero().squeeze().cpu().numpy().flatten().tolist()])

            log(f'prob: {prob[0]:.7f}\ncomet_prob: {self.model.comet_prob:.7f}')
            log(f'tmpl: {tmpl_pred_log}\ntmpl_gt: {tmpl_gt_log}\ntmpl_entropy: {tmpl_entropy:.7f}')
            log(f'obj1: {obj1_pred_log}\nobj2: {obj2_pred_log}\nobjk_gt: {objk_gt_log}')

            if self.params['do_comm_expl']:
                log(f'root_node: {self.model.root_node[0]}')
                log(f'root_node_person2you: {self.model.root_node_person2you[self.model.root_node[0]]}')

            # Log all_score_df
            act_name = 'tmpl'
            if all_score_df is not None:
                columns = [f'act4comet', 'tmpl_token', 'agt2tmpl_score',
                           'obj1_token', 'agt2obj1_score', 'obj2_token', 'agt2obj2_score',
                           'head', 'edge', 'tail', 'edge4act',
                           'agt2act_score', 'node2act_score', 'node2node_score', 'total_score', 'prob']
                values = all_score_df.loc[all_score_df.loc[:, 'batch'] == 0, [f'{act_name}_token', 'total_score']]
                values = all_score_df.loc[values.groupby(f'{act_name}_token').idxmax().loc[:, 'total_score'], columns]
                values = values.sort_values(by=['total_score'], ascending=False).values.tolist()
                values = [[f'{c}, {v}' if type(v) == str else f'{c}, {round(v, 6)}' for c, v in zip(columns, value)] for value in values]
                tb.logkv_mean('TotalScore', float(values[0][-1].split(', ')[-1]))
                log('\n'.join([' & '.join(value) for value in values]))

            # Choose next action & step()
            chosen_act = self.decode_actions(tmpl_token_id, obj1_token_id, obj2_token_id)
            obs, rewards, dones, infos, graph_infos = self.vec_env.step(chosen_act)

            # Log
            tb.logkv('Step', step)                                       # tensorboard
            tb.logkv_mean('Value', state_value.mean().item())                  # tensorboard
            tb.logkv_mean('TotalStepsPerEpisode', sum([i['steps'] for i in infos]) / float(len(graph_infos)))
            tb.logkv_mean('Valid', infos[0]['valid'])
            log(f'Act: {chosen_act[0]} => Rew {rewards[0]}, Score {infos[0]["score"]}, Done {dones[0]}, Value {state_value[0].item():.3f}')
            log('Obs: {}'.format(clean(obs[0])))
            if dones[0]:
                log('Step {} EpisodeScore {}\n'.format(step, infos[0]['score']))
            for done, info in zip(dones, infos):
                if done:
                    tb.logkv_mean('EpisodeScore', info['score'])

            #
            rew_tt = torch.FloatTensor(rewards).to(device).unsqueeze(1)
            done_mask_tt = (~torch.tensor(dones)).float().to(device).unsqueeze(1)
            self.model.reset_hidden(done_mask_tt)
            transitions.append((
                state_value, rew_tt, done_mask_tt, tmpl, tmpl_gt, obj1, obj2, obj_mask_gt, graph_mask
            ))

            if len(transitions) >= self.params['bptt']:
                tb.logkv('StepsPerSecond', float(step) / (time.time() - start))
                self.model.clone_hidden()
                obs_reps = np.array([g.ob_rep for g in graph_infos])
                graph_mask = self.generate_graph_mask(graph_infos)
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                scores = [info['score'] for info in infos]
                adm_tmpl = [[a.template_id for a in g.admissible_actions] for g in graph_infos]

                next_value, _, _, _, _, _, _ = self.model(
                    obs_reps, scores, graph_state_reps, graph_mask, adm_tmpl, test=True
                )
                returns, advantages = self.discount_reward(transitions, next_value)
                log('Returns: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in returns]))
                log('Advants: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in advantages]))
                tb.logkv_mean('Advantage', advantages[-1].median().item())
                loss = self.update(transitions, returns, advantages).clone()
                del transitions[:]
                self.model.restore_hidden()

            if step % 5000 == 0:
                torch_save_file = os.path.join(self.output_dir, f'kga2c-{step}.pt')
                torch.save(self.model.state_dict(), torch_save_file)
                torch.save(self.model.entropy_list, os.path.join(self.output_dir, f'saved_entropy_data-{step}.pt'))
                print(f'===== Saved the model in {torch_save_file} at {step} steps =====')

                if self.params['store_comet_output'] is not None:
                    from pathlib import Path
                    cskg_dir = f'{self.params["store_comet_output"]}/cskg/{self.params["env_name"]}'
                    cskg_dir_size = sum(f.stat().st_size for f in Path(cskg_dir).glob('**/*') if f.is_file())
                    print(f'===== {cskg_dir} has size of {cskg_dir_size} bytes =====')

            if step % self.params['checkpoint_interval'] == 0:
                torch_save_file = os.path.join(self.output_dir, 'kga2c.pt')
                torch.save(self.model.state_dict(), torch_save_file)
                torch.save(self.model.entropy_list, os.path.join(self.output_dir, 'saved_entropy_data.pt'))
                print(f'===== Saved the model in {torch_save_file} at {step} steps =====')

                act_name = 'tmpl'
                if all_score_df is not None:
                    csv_name = os.path.join(
                        self.output_dir, 'cskg',
                        f'all_score_df_{act_name}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}.csv'
                    )
                    all_score_df.sort_values(by=['batch', f'{act_name}_token_id', 'tail', 'edge4act']) \
                        .to_csv(csv_name, index=False)
                    print(f'===== Saved the all_score_df in {csv_name} at {step} steps =====')

                if cskg_df is not None:
                    csv_name = os.path.join(
                        self.output_dir, 'cskg',
                        f'cskg_df-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}.csv'
                    )
                    cskg_df.to_csv(csv_name, index=False)
                    print(f'===== Saved the cskg_df in {csv_name} at {step} steps =====')

                log(f'entropy_list_size: {self.model.entropy_list_size}')
                log(f'entropy_list: {self.model.entropy_list[0]}')
                log(f'tmpl_min_prob: {self.model.tmpl_min_prob:.7f}')

        self.vec_env.close_extras()

    def update(self, transitions, returns, advantages):
        assert len(transitions) == len(returns) == len(advantages)
        loss = 0
        for trans, ret, adv in zip(transitions, returns, advantages):
            state_value, rew_tt, _, tmpl, tmpl_gt, obj1, obj2, objk_mask_gt, graph_mask = trans
            tmpl_token_id, tmpl_distribution, obj_num_in_tmpl = tmpl
            objk_token_id, objk_distribution = list(zip(obj1, obj2))
            objk_token_id, objk_distribution = torch.stack(objk_token_id), torch.stack(objk_distribution)

            # Supervised Template Loss
            tmpl_probs = F.softmax(tmpl_distribution, dim=1)
            template_loss = self.params['template_coeff'] * self.loss_fn1(tmpl_probs, tmpl_gt)

            # Supervised Object Loss
            object_mask_target = objk_mask_gt.permute((1, 0, 2))
            obj_probs = F.softmax(objk_distribution, dim=2)
            object_mask_loss = self.params['object_coeff'] * self.loss_fn1(obj_probs, object_mask_target)

            # Build the object mask
            o1_mask, o2_mask = [0] * self.batch_size, [0] * self.batch_size
            for d, st in enumerate(obj_num_in_tmpl):
                if st > 1:
                    o1_mask[d] = 1
                    o2_mask[d] = 1
                elif st == 1:
                    o1_mask[d] = 1
            o1_mask = torch.FloatTensor(o1_mask).to(device)
            o2_mask = torch.FloatTensor(o2_mask).to(device)

            # policy gradient loss on obj
            policy_obj_loss = torch.FloatTensor([0]).to(device)
            cnt = 0
            for i in range(self.batch_size):
                if obj_num_in_tmpl[i] >= 1:
                    cnt += 1
                    batch_pred = objk_distribution[0, i, graph_mask[i]]
                    action_log_probs_obj = F.log_softmax(batch_pred, dim=0)
                    dec_obj_idx = objk_token_id[0, i].item()
                    graph_mask_list = graph_mask[i].nonzero().squeeze().cpu().numpy().flatten().tolist()
                    idx = graph_mask_list.index(dec_obj_idx)
                    log_prob_obj = action_log_probs_obj[idx]
                    policy_obj_loss += -log_prob_obj * adv[i].detach()
            if cnt > 0:
                policy_obj_loss /= cnt
            log_probs_obj = F.log_softmax(objk_distribution, dim=-1)
            tb.logkv_mean('PolicyObjLoss', policy_obj_loss.item())

            # policy gradient loss on tmpl
            log_probs_tmpl = F.log_softmax(tmpl_distribution, dim=-1)
            action_log_probs_tmpl = log_probs_tmpl.gather(1, tmpl_token_id).squeeze()
            policy_tmpl_loss = (-action_log_probs_tmpl * adv.detach().squeeze()).mean()
            tb.logkv_mean('PolicyTemplateLoss', policy_tmpl_loss.item())

            policy_loss = policy_tmpl_loss + policy_obj_loss

            value_loss = self.params['value_coeff'] * self.loss_fn3(state_value, ret)
            tmpl_entropy = -(tmpl_probs * log_probs_tmpl).mean()
            tb.logkv_mean('TemplateEntropy', tmpl_entropy.item())
            object_entropy = -(obj_probs * log_probs_obj).mean()
            tb.logkv_mean('ObjectEntropy', object_entropy.item())
            # Minimizing entropy loss will lead to increased entropy
            entropy_loss = self.params['entropy_coeff'] * -(tmpl_entropy + object_entropy)

            loss += template_loss + object_mask_loss + value_loss + entropy_loss + policy_loss

        tb.logkv('Loss', loss.item())
        tb.logkv('TemplateLoss', template_loss.item())
        tb.logkv('ObjectLoss', object_mask_loss.item())
        tb.logkv('PolicyLoss', policy_loss.item())
        tb.logkv('ValueLoss', value_loss.item())
        tb.logkv('EntropyLoss', entropy_loss.item())
        tb.dumpkvs()
        # template_loss.backward(retain_graph=True)
        # object_mask_loss.backward(retain_graph=True)
        # value_loss.backward(retain_graph=True)
        # entropy_loss.backward(retain_graph=True)
        # policy_loss.backward()

        loss.backward()

        # Compute the gradient norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm += p.grad.data.norm(2).item()
        tb.logkv('UnclippedGradNorm', grad_norm)

        nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])

        # Clipped Grad norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm += p.grad.data.norm(2).item()
        tb.logkv('ClippedGradNorm', grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def decode_actions(self, tmpl_token_id, obj1_token_id, obj2_token_id):
        '''
        Returns string representations of the given template actions.

        :param decoded_template: Tensor of template indices.
        :type decoded_template: Torch tensor of size (Batch_size x 1).
        :param decoded_objects: Tensor of o1, o2 object indices.
        :type decoded_objects: Torch tensor of size (2 x Batch_size x 1).

        '''
        decoded_actions = []
        for i in range(self.batch_size):
            decoded_template = tmpl_token_id[i].item()
            decoded_object1 = obj1_token_id[i].item()
            decoded_object2 = obj2_token_id[i].item()
            decoded_action = self.tmpl_to_str(decoded_template, decoded_object1, decoded_object2)
            decoded_actions.append(decoded_action)
        return decoded_actions

    def tmpl_to_str(self, template_idx, o1_id, o2_id):
        """ Returns a string representation of a template action. """
        template_str = self.template_generator.templates[template_idx]
        holes = template_str.count('OBJ')
        assert holes <= 2
        if holes <= 0:
            return template_str
        elif holes == 1:
            return template_str.replace('OBJ', self.vocab_act[o1_id])
        else:
            return template_str.replace('OBJ', self.vocab_act[o1_id], 1) \
                .replace('OBJ', self.vocab_act[o2_id], 1)

    def test(self):
        with torch.no_grad():
            print("=== === === start testing!!! === === ===")
            obs, infos, graph_infos = self.vec_env.reset()  # "class KGA2CEnv" in env.py
            done_step, done_indicator = 0, False
            for step in range(1, 10000 + 1):
                # model()
                # obs_reps[:, (0, 1, 2, 3), :] = (room description, inventory, game feedback, previous action)
                obs_reps = np.array([g.ob_rep for g in graph_infos])         # obs_rep
                scores = [info['score'] for info in infos]                   # score
                graph_state_reps = [g.graph_state_rep for g in graph_infos]  # graph_state_reps
                graph_mask = self.generate_graph_mask(graph_infos)           # masking on kg/interactive/none options
                adm_tmpl = [[a.template_id for a in g.admissible_actions] for g in graph_infos]

                # model
                state_value, tmpl, obj1, obj2, prob, cskg_df, all_score_df = self.model(
                    obs_reps, scores, graph_state_reps, graph_mask, adm_tmpl
                )
                tmpl_token_id, tmpl_distribution, obj_num_in_tmpl = tmpl
                obj1_token_id, obj1_distribution = obj1
                obj2_token_id, obj2_distribution = obj2

                # Generate the ground truth and object mask
                admissible = [g.admissible_actions for g in graph_infos]
                objs = [g.objs for g in graph_infos]
                tmpl_gt, obj_mask_gt = self.generate_targets(admissible, objs)

                # Log template predictions (tmpl_gt) and ground truth in graph (obj_mask_gt)
                tmpl_distribution_0 = F.softmax(tmpl_distribution[0].detach().clone(), dim=-1) + torch.tensor(1e-7)
                tmpl_entropy = -(tmpl_distribution_0 * tmpl_distribution_0.log()).mean().item()
                tmpl_prob_topk, tmpl_id_topk = tmpl_distribution_0.topk(20)
                tmpl_topk = [self.template_generator.templates[t] for t in tmpl_id_topk.tolist()]
                # [TEMPL] [PROBABILITY], [TEMPL] [PROBABILITY], ...
                tmpl_pred_log = ', '.join([f'{_a} {round(_b, 3)}' for _a, _b in zip(tmpl_topk, tmpl_prob_topk.tolist())])
                tmpl_gt_log = ', '.join([self.template_generator.templates[i] for i in tmpl_gt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()])

                # Log object predictions (obj_distribution) and ground truth in graph (obj_mask_gt)
                objk_prob_topk, objk_id_topk = F.softmax(torch.stack([obj1_distribution[0], obj2_distribution[0]]), dim=-1).topk(5)
                obj1_id_topk, obj2_id_topk = [[self.vocab_act[_ij] for _ij in _i] for _i in objk_id_topk.tolist()]
                # [OBJ] [PROBABILITY], [OBJ] [PROBABILITY], ...
                obj1_pred_log = ', '.join(['{} {:.3f}'.format(o, o_prob) for o, o_prob in zip(obj1_id_topk, objk_prob_topk[0].tolist())])
                obj2_pred_log = ', '.join(['{} {:.3f}'.format(o, o_prob) for o, o_prob in zip(obj2_id_topk, objk_prob_topk[1].tolist())])
                objk_gt_log = ', '.join([self.vocab_act[i] for i in obj_mask_gt[0, 0].nonzero().squeeze().cpu().numpy().flatten().tolist()])

                log(f'prob: {prob[0]:.7f}\ncomet_prob: {self.model.comet_prob:.7f}')
                log(f'tmpl: {tmpl_pred_log}\ntmpl_gt: {tmpl_gt_log}\ntmpl_entropy: {tmpl_entropy:.7f}')
                log(f'obj1: {obj1_pred_log}\nobj2: {obj2_pred_log}\nobjk_gt: {objk_gt_log}')

                if self.params['do_comm_expl']:
                    log(f'root_node: {self.model.root_node[0]}')
                    log(f'root_node_person2you: {self.model.root_node_person2you[self.model.root_node[0]]}')

                # Log all_score_df
                act_name = 'tmpl'
                if all_score_df is not None:
                    columns = [f'act4comet', 'tmpl_token', 'agt2tmpl_score',
                               'obj1_token', 'agt2obj1_score', 'obj2_token', 'agt2obj2_score',
                               'head', 'edge', 'tail', 'edge4act',
                               'agt2act_score', 'node2act_score', 'node2node_score', 'total_score', 'prob']
                    values = all_score_df.loc[all_score_df.loc[:, 'batch'] == 0, [f'{act_name}_token', 'total_score']]
                    values = all_score_df.loc[values.groupby(f'{act_name}_token').idxmax().loc[:, 'total_score'], columns]
                    values = values.sort_values(by=['total_score'], ascending=False).values.tolist()
                    values = [[f'{c}, {v}' if type(v) == str else f'{c}, {round(v, 6)}' for c, v in zip(columns, value)] for value in values]
                    tb.logkv_mean('TotalScore', float(values[0][-1].split(', ')[-1]))
                    log('\n'.join([' & '.join(value) for value in values]))

                # Choose next action & step()
                chosen_act = self.decode_actions(tmpl_token_id, obj1_token_id, obj2_token_id)
                obs, rewards, dones, infos, graph_infos = self.vec_env.step(chosen_act)

                # Log
                tb.logkv('Step', step)                                       # tensorboard
                tb.logkv_mean('Value', state_value.mean().item())                  # tensorboard
                tb.logkv_mean('TotalStepsPerEpisode', sum([i['steps'] for i in infos]) / float(len(graph_infos)))
                tb.logkv_mean('Valid', infos[0]['valid'])
                log(f'Act: {chosen_act[0]} => Rew {rewards[0]}, Score {infos[0]["score"]}, Done {dones[0]}, Value {state_value[0].item():.3f}')
                log('Obs: {}'.format(clean(obs[0])))
                if dones[0]:
                    log('Step {} EpisodeScore {}\n'.format(step, infos[0]['score']))
                for done, info in zip(dones, infos):
                    if done:
                        tb.logkv_mean('EpisodeScore', info['score'])
                        done_indicator = True

                #
                rew_tt = torch.FloatTensor(rewards).to(device).unsqueeze(1)
                done_mask_tt = (~torch.tensor(dones)).float().to(device).unsqueeze(1)
                self.model.reset_hidden(done_mask_tt)

                if step % self.params['bptt'] == 0:
                    if done_indicator is True:
                        done_step += 1
                    done_indicator = False
                    tb.logkv_mean('done_step', done_step)
                    tb.dumpkvs()
                    if done_step > 100:
                        break

            self.vec_env.close_extras()

# if __name__ == "__main__":
#     KGA2CTrainer
