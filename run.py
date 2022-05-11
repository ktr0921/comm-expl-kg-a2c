import os
import argparse
from datetime import datetime

from trainer.trainer import KGA2CTrainer

# 905 acorncourt advent adventureland anchor awaken dragon deephome karn moonlit temple yomomma
env_name_list = [
    '905', 'acorncourt', 'advent', 'adventureland', 'afflicted', 'anchor', 'awaken', 'balances', 'ballyhoo', 'curses',
    'cutthroat', 'deephome', 'detective', 'dragon', 'enchanter', 'enter', 'gold', 'hhgg', 'hollywood', 'huntdark',
    'infidel', 'inhumane', 'jewel', 'karn', 'lgop', 'library', 'loose', 'lostpig', 'ludicorp', 'lurking', 'moonlit',
    'murdac', 'night', 'omniquest', 'partyfoul', 'pentari', 'planetfall', 'plundered', 'reverb', 'seastalker',
    'sherlock', 'snacktime', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'theatre', 'trinity', 'tryst205', 'weapon',
    'wishbringer', 'yomomma', 'zenon', 'zork1', 'zork2', 'zork3', 'ztuu'
]


def parse_args():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--env_name', default='zork1', choices=env_name_list)
    parser.add_argument('--output_dir', default='outputs/')
    parser.add_argument('--output_name', default='')
    parser.add_argument('--spm_file', default='etc/spm_models/unigram_8k.model')
    parser.add_argument('--openie_path', default='etc/stanford-corenlp-full-2018-10-05')
    parser.add_argument('--openie_port', default=None, type=int)
    parser.add_argument('--redis_path', default='/usr/local/Cellar/redis/6.2.1/bin/redis-server')
    parser.add_argument('--redis_config_path', default=None, type=str)
    parser.add_argument('--redis_port', default=6970, type=int)
    #
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--gamma', default=.5, type=float)
    parser.add_argument('--embedding_size', default=50, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--padding_idx', default=0, type=int)
    parser.add_argument('--gat_emb_size', default=50, type=int)
    parser.add_argument('--dropout_ratio', default=0.2, type=float)
    parser.add_argument('--preload_weights', default='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--steps', default=25000, type=int)
    parser.add_argument('--reset_steps', default=100, type=int)
    parser.add_argument('--stuck_steps', default=10, type=int)
    parser.add_argument('--trial', default='base')
    parser.add_argument('--loss', default='value_policy_entropy')
    parser.add_argument('--graph_dropout', default=0.0, type=float)
    parser.add_argument('--k_object', default=1, type=int)
    parser.add_argument('--g_val', default=False, type=bool)
    parser.add_argument('--entropy_coeff', default=0.03, type=float)
    parser.add_argument('--clip', default=40, type=int)
    parser.add_argument('--bptt', default=8, type=int)
    parser.add_argument('--value_coeff', default=9, type=float)
    parser.add_argument('--template_coeff', default=3, type=float)
    parser.add_argument('--object_coeff', default=9, type=float)
    parser.add_argument('--recurrent', default=True, type=bool)
    parser.add_argument('--checkpoint_interval', default=500, type=int)
    parser.add_argument('--no-gat', dest='gat', action='store_false')
    parser.add_argument('--masking', default='kg', choices=['kg', 'interactive', 'none'], help='Type of object masking applied')
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #kelvin
    # comet
    parser.add_argument('--do_comm_expl', action='store_true')
    parser.add_argument('--comet_path', default='etc/pre-trained-models/COMeT-atomic_2020_BART', type=str)
    parser.add_argument('--cskg_size', default=1, type=int)
    parser.add_argument('--num_generate', default=2, type=int)
    parser.add_argument('--store_comet_output', default='etc', type=str)
    # sample action
    parser.add_argument('--tmpl_max_number', default=7, type=int)
    parser.add_argument('--tmpl_min_prob', default=0.75, type=float)
    parser.add_argument('--include_adm_act', action='store_true')
    parser.add_argument('--act_sampling_type', default='softmax', type=str, choices=['max', 'softmax'])
    # comet scheduler
    parser.add_argument('--comet_prob', default=0.5, type=float)
    parser.add_argument('--do_entropy_threshold', action='store_true')
    parser.add_argument('--entropy_list_size', default=1000, type=int)

    parser.add_argument('--comet_start_prob', default=0.5, type=float)
    parser.add_argument('--comet_start_step', default=30000, type=int)
    parser.add_argument('--comet_end_prob', default=0.5, type=float)
    parser.add_argument('--comet_end_step', default=45000, type=int)
    # relations
    edge_sampling_type = ['all', 'random']
    parser.add_argument('--vv_edge', default='xIntent,xNeed', type=str)
    parser.add_argument('--va_edge', default='xEffect,xWant', type=str)
    parser.add_argument('--vv_edge_sampling_n', default=0, type=int)
    parser.add_argument('--va_edge_sampling_n', default=0, type=int)
    parser.add_argument('--vv_edge_sampling_type', default='all', type=str, choices=edge_sampling_type)
    parser.add_argument('--va_edge_sampling_type', default='all', type=str, choices=edge_sampling_type)
    # gamma
    parser.add_argument('--vv_prev_score_gamma', default=0.3, type=float)
    parser.add_argument('--vv_score_gamma',      default=1.0, type=float)
    parser.add_argument('--va_score_gamma',      default=0.7, type=float)
    parser.add_argument('--tmpl_score_gamma',    default=1.0, type=float)
    parser.add_argument('--aa_score_gamma',      default=0.8, type=float)
    # test
    parser.add_argument('--test_first_100', action='store_true')
    # parser.add_argument('--cuda', default=None, type=str)
    parser.add_argument('--test', default='', type=str)
    # test_list = ['cskg', 'va_score', 'score', 'prob', 'entropy_threshold', 'score_threshold', 'total_score']
    # parser.add_argument('--test', default='', type=str, choices=test_list)
    # # penalty if repeating
    # parser.add_argument('--no_repeating_act_n', default=0, type=int)
    # parser.add_argument('--no_repeating_act_reward', default=1, type=float)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    parser.set_defaults(gat=True)
    parser.set_defaults(include_adm_act=True)
    args = parser.parse_args()
    params = vars(args)
    return params


if __name__ == "__main__":
    params = parse_args()

    params['bindings'] = params['env_name']
    params['tsv_file'] = os.path.join('etc', 'data', params['env_name'] + '_entity2id.tsv')
    dir_roms = os.path.join('etc', 'roms')
    rom_file_path = [os.path.join(dir_roms, rom) for rom in os.listdir(dir_roms)
                     if params['env_name'] == rom.rsplit('.', 1)[0]]
    params['rom_file_path'] = rom_file_path[0]

    tmp = params['env_name'] + '-' + datetime.now().strftime("%Y%m%d_%H%M%S") + '-' + params['output_name']
    params['output_dir'] = os.path.join(params['output_dir'], tmp.strip('-'))

    params['openie_port'] = params["redis_port"] + 2040 \
        if params['openie_port'] is None else params['openie_port']
    params['redis_config_path'] = f'etc/redis-conf/redis{params["redis_port"]}.conf' \
        if params['redis_config_path'] is None else params['redis_config_path']
    print(params)
    trainer = KGA2CTrainer(params)
    if params['test_first_100'] is False:
        trainer.train(params['steps'])
    else:
        trainer.test()
