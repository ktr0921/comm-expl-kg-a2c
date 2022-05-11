import json
import argparse
import itertools
from tqdm import tqdm
from pathlib import Path
import pandas as pd

import torch
import torch.nn.functional as F

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch

from logging import getLogger
logger = getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
]


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(self, queries, decode_method="beam", num_generate=5, ):
        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)
            return decs

    def generate_with_scores(self, queries, num_generate=5, min_length=0, max_length=24, n_square=True, ):
        with torch.no_grad():
            examples = queries

            decs, sequences_scores = [], []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate*3,
                    num_return_sequences=num_generate,
                    output_scores=True,
                    return_dict_in_generate=True,
                    min_length=min_length,
                    max_length=max_length,
                )

                if num_generate == 1:
                    if n_square:
                        sequences_scores_i = torch.stack(summaries.scores).softmax(-1).topk(1).values.log().sum() / \
                                             (summaries.sequences.size(-1) ** 2)
                    else:
                        sequences_scores_i = torch.stack(summaries.scores).softmax(-1).topk(1).values.log().sum() / \
                                             (summaries.sequences.size(-1))
                    sequences_scores_i = sequences_scores_i.unsqueeze(-1)
                    sequences_scores.append(sequences_scores_i)
                else:
                    sequences_scores.append(summaries.sequences_scores)

                dec = self.tokenizer.batch_decode(summaries.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)
            return decs, sequences_scores

    # def generate_with_scores2(self, queries, num_generate=5, min_length=0, max_length=24, n_square=True, ):
    #     with torch.no_grad():
    #         examples = queries
    #
    #         decs, sequences_scores = [], []
    #         for batch in list(chunks(examples, self.batch_size)):
    #
    #             batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(device)
    #             input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)
    #
    #             summaries = self.model.generate(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 decoder_start_token_id=self.decoder_start_token_id,
    #                 num_beams=num_generate*3,
    #                 num_return_sequences=num_generate,
    #                 output_scores=True,
    #                 return_dict_in_generate=True,
    #                 output_hidden_states=True,
    #                 min_length=min_length,
    #                 max_length=max_length,
    #             )
    #
    #             return summaries

            #     if num_generate == 1:
            #         if n_square:
            #             sequences_scores_i = torch.stack(summaries.scores).softmax(-1).topk(1).values.log().sum() / \
            #                                  (summaries.sequences.size(-1) ** 2)
            #         else:
            #             sequences_scores_i = torch.stack(summaries.scores).softmax(-1).topk(1).values.log().sum() / \
            #                                  (summaries.sequences.size(-1))
            #         sequences_scores_i = sequences_scores_i.unsqueeze(-1)
            #         sequences_scores.append(sequences_scores_i)
            #     else:
            #         sequences_scores.append(summaries.sequences_scores)
            #
            #     dec = self.tokenizer.batch_decode(summaries.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            #     decs.append(dec)
            # return decs, sequences_scores


relation_to_readable_phrase = {
    'xNeed': 'but before, PersonX needed',
    'xEffect': 'as a result, PersonX will',
    'xIntent': 'because PersonX wanted',
    'xReason': 'because',
    'xWant': 'as a result, PersonX wants',
}

pyhsical_entity = [
    'ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty', 'CapableOf', 'Desires', 'NotDesires'
]
event_centered = [
    'isAfter', 'HasSubEvent', 'IsBefore', 'HinderedBy', 'Causes', 'xReason', 'isFilledBy',
]
social_interaction = [
    'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant',
]

if __name__ == "__main__":
    # sample usage
    print("model loading ...")
    comet = Comet("etc/pre-trained-models/comet-atomic_2020_BART")
    comet.model.zero_grad()
    print("model loaded")

    # ==========================================================================================

    queries = []

    head = [
        # 'Canyon Bottom. PersonX is beneath the walls of the river canyon which may be climbable here. The lesser part of the runoff of Aragain Falls flows by below. To the north is a narrow path.',
        # 'PersonX wants to clean up the room',
        # 'Kitchen. PersonX is in the kitchen of the white house. A table seems to have been used recently for the preparation of food. A passage leads to the west and a dark staircase can be seen leading upward. A dark chimney leads down and to the east is a small window which is open. On the table is an elongated brown sack, smelling of hot peppers. A bottle is sitting on the table. The glass bottle contains. A quantity of water.',
        # 'PersonX go west. Kitchen. PersonX is in the kitchen of the white house. A table seems to have been used recently for the preparation of food. A passage leads to the west and a dark staircase can be seen leading upward. A dark chimney leads down and to the east is a small window which is open. On the table is an elongated brown sack, smelling of hot peppers. A bottle is sitting on the table. The glass bottle contains. A quantity of water.',
        # 'Living Room. PersonX is in the living room. There is a doorway to the east, a wooden door with strange gothic lettering to the west, which appears to be nailed shut, a trophy case, and a closed trap door at PersonXr feet. Above the trophy case hangs an elvish sword of great antiquity. A batterypowered brass lantern is on the trophy case.',
        'West of House. PersonX is standing in an open field west of a white house, with a boarded front door. There is a small mailbox here.',
    ]
    # , 'xEffect', 'xWant'
    rel = [
        'xNeed', 'xIntent', 'xWant',
    ]
    for h in head:
        for r in rel:
            queries.append("{} {} [GEN]".format(h, r))

    num_generate = 2

    print(queries)

    # results = COMeT.generate(queries, decode_method="beam", num_generate=5)
    decs, sequences_scores = comet.generate_with_scores(
        queries, num_generate=num_generate
    )
    print('===')
    print(rel)
    print(decs)
    print(sequences_scores)
    print('===')
    print('===')
    print('===')

    # ==========================================================================================

    head = [
        'West of House. PersonX is standing in an open field west of a white house, with a boarded front door. There is a small mailbox here.',
    ]

    rel = [
        'xWant',
    ]
    queries = []
    for h in head:
        for r in rel:
            queries.append("{} {} [GEN]".format(h, r))

    act_all = {'open mailbox'}
    act = {'</s> to ' + act_i.strip() for act_i in act_all}

    print(f'queries: {queries}')
    print(f'act: {act}')

    start = 2

    input_token = comet.tokenizer(queries, return_tensors="pt", truncation=True, padding="max_length")
    input_ids, att_mask = trim_batch(**input_token.to(device), pad_token_id=comet.tokenizer.pad_token_id)

    for act_i in act:
        output_ids = comet.tokenizer(act_i, return_tensors="pt", add_special_tokens=False).input_ids
        output_ids = output_ids.to(device)

        # iterate to get log likelihood of each token after '</s> PersonX' or '</s> to'
        trg_token_log_prob = []

        print('===')

        print(input_ids)
        print(att_mask)
        print(output_ids)

        print('===')

        print(input_ids[0])
        print(att_mask[0])
        print(output_ids[0])
        print(output_ids[:, :3])
        print(output_ids.size(1))

        print('===')

        for i in range(start, output_ids.size(1)):
            generated_outputs = comet.model.generate(
                input_ids=input_ids,
                attention_mask=att_mask,
                decoder_start_token_id=output_ids[:, :i],
                max_length=i + 1,
                num_beams=1,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            print(generated_outputs)

            # get log likelihood
            next_token_dist = generated_outputs.scores[0].softmax(-1)
            trg_token_log_prob_i = (next_token_dist[:, output_ids[:, i].item()] + torch.tensor(1e-7)).log()
            trg_token_log_prob.append(trg_token_log_prob_i)
        n_square = False
        trg_token_log_prob = torch.stack(trg_token_log_prob).sum(0) / (len(trg_token_log_prob) ** 2) \
            if n_square else torch.stack(trg_token_log_prob).sum(0) / len(trg_token_log_prob)

        print(trg_token_log_prob)

    # ==========================================================================================

    # # get node2act score
    # va_score_df = pd.DataFrame(rows, columns=['node', 'edge4act', 'act', 'node2act_score'])
    # for act_i, triple_missing_i in zip(act, triple_missing):
    #     # tokenize act
    #
    #     # iterate to get log likelihood of each token after '</s> PersonX' or '</s> to'
    #     trg_token_log_prob = []
    #     if sum(triple_missing_i) != 0:
    #         for i in range(start, output_ids.size(1)):
    #             generated_outputs = self.COMeT.model.generate(
    #                 input_ids=input_ids[triple_missing_i],
    #                 attention_mask=att_mask[triple_missing_i],
    #                 decoder_start_token_id=output_ids[:, :i],
    #                 max_length=i + 1,
    #                 num_beams=1,
    #                 num_return_sequences=1,
    #                 output_scores=True,
    #                 return_dict_in_generate=True,
    #                 do_sample=False,
    #             )
    #
    #             # get log likelihood
    #             next_token_dist = generated_outputs.scores[0].softmax(-1)
    #             trg_token_log_prob_i = (next_token_dist[:, output_ids[:, i].item()] + torch.tensor(1e-7)).log()
    #             trg_token_log_prob.append(trg_token_log_prob_i)
    #
    #         # calculate avg lod likelihood
    #         trg_token_log_prob = torch.stack(trg_token_log_prob).sum(0) / (len(trg_token_log_prob) ** 2) \
    #             if n_square else torch.stack(trg_token_log_prob).sum(0) / len(trg_token_log_prob)
    #
    #         if self.store_comet_output is not None:
    #             node2rows_j = [node2rows[j] for j, _ij in enumerate(triple_missing_i) if _ij]
    #             edge2rows_j = [edge2rows[j] for j, _ij in enumerate(triple_missing_i) if _ij]
    #             for head_j, va_edge_jk, log_prob_i in zip(node2rows_j, edge2rows_j, trg_token_log_prob.tolist()):
    #                 head_tmp = head_j.replace(".", " ").split()[::3]
    #                 k = int((len(head_tmp) - 1) / 2) if len(head_tmp) < 10 else 5
    #                 triple = f'{self.store_comet_output}/cskg/{self.params["env_name"]}/{"".join(head_tmp[:k] + head_tmp[-k:])}/{va_edge_jk}/va'
    #
    #                 with open(triple, 'a') as f:
    #                     f.write(f'\n\n{head_j}\n{act_i}\t{log_prob_i}')
    #
    #         else:
    #             node2rows_j = node2rows
    #             edge2rows_j = edge2rows
    #
    #         # append the results
    #         rows = list(zip(
    #             node2rows_j, edge2rows_j, [' '.join(act_i.split()[2:])] * len(node2rows_j), trg_token_log_prob.tolist()
    #         ))
    #         rows_df = pd.DataFrame(rows, columns=va_score_df.columns)
    #         va_score_df = va_score_df.append(rows_df, ignore_index=True)
    #
    #     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #kelvin
    #     if self.params['test'] == 'va_score':  # kelvin test
    #         print('===')
    #         print('=== generate_va_score triple_missing_i')
    #         print(f'"{triple_missing_i}"')
    #         print('=== generate_va_score trg_token_log_prob')
    #         print(f'"{trg_token_log_prob}"')
    #         print('===')
    #     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

