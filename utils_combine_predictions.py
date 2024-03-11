import logging
# logging.basicConfig(level=logging.ERROR)

from copy import deepcopy
from pprint import pprint
import spacy
import json, jsonlines
import logging
from collections import Counter
import argparse

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Spacy NLP initialization
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_json_data(json_file_path):
    """
    Read thebest_predictions_valid.json
    """
    json_data = []
    with open(json_file_path, "r", encoding="utf-8") as reader:
        json_data = json.load(reader)
    logging.info(f'[Loaded thebest_predictions_.json] with {len(json_data)} examples. It includes the pronouns in the same sample. ')
    return json_data

def read_augwow_examples(input_file):
    """
    Read AugWoW examples with the found_pronoun.
    """
    examples = []
    with open(input_file, "r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader):
            item = json.loads(line)
            examples.append(item)
    logging.info(f'Loaded {len(examples)} examples with pronouns found. It includes the pronouns in the same sample.')
    return examples

def read_dialfact_examples(input_file):
    """
    Read Dialfact examples with the found_pronoun.
    """
    examples = []
    with open(input_file, "r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader):
            item = json.loads(line)
            examples.append(item)
    logging.info(f'Loaded {len(examples)} examples with pronouns found. It includes the pronouns in the same sample.')
    return examples

def write_jsonl(data, file_path):
    logging.info(f'Writing {len(data)} items to {file_path}.')
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def find_sample(examples, qas_id):
    for s in examples:
        if s['qas_id'] == qas_id:
            return s['item']
        
def words_to_response(task, sample):
    """
    Input: sample with dict_words 
    Output: sample with response
    """
    new_sample = deepcopy(sample)
    words = sample['words'] # key: token_idx, value: word (in response)
    words = [str(token) for token in list(words.values())]
    if task == 'dialfact':
        new_sample['response'] = ' '.join(words)
    elif task == 'augwow':
        new_sample['claim'] = sample['claim'].split("[RESPONSE]:")[0] + "[RESPONSE]: " + ' '.join(words)
    return new_sample

def combine_predictions(args, examples, thebest_data, output_file):#, preprocess_file):
    results = {} # key: qas_id, value: item
    
    cnt_all = 0
    cnt_dot = 0
    cnt_empty = 0
    for qas_id, pred_noun in thebest_data.items():

        sample_id = qas_id[:qas_id[:qas_id.rfind('_')].rfind('_')]
        pronoun_idx = qas_id[qas_id.rfind('_')+1:]
        # print(qas_id, '--->', sample_id)
        # print(qas_id, '--->', pronoun_idx)
        if sample_id in results: # 이미 results안에 sample이 있는 경우
            sample = results[sample_id]
            if 'empty' in pred_noun[0]:
                cnt_empty += 1
                continue
            elif '.' == pred_noun[0]:
                cnt_dot += 1
                continue
            sample['words'][int(pronoun_idx)] = pred_noun[0]
        else: # results안에 sample이 없는 경우, 처음으로 등장하는 sample인 경우
            sample = find_sample(examples, qas_id)
            
            ori_response = ''
            if args.task == 'dialfact':
                ori_response = sample['response']
            elif args.task == 'augwow':
                ori_response = sample['claim'].split('[RESPONSE]:')[-1].lstrip()
            
            dict_words = {i: item for i, item in enumerate([token.text for token in nlp(ori_response)])}
            if 'empty' in pred_noun[0]:
                cnt_empty += 1
                continue
            elif '.' == pred_noun[0]:
                cnt_dot += 1
                continue
            dict_words[int(pronoun_idx)] = pred_noun[0]
            sample['words'] = dict_words
            results[sample_id] = sample
        
        cnt_all += 1
        # if cnt==10: break ###### for testing

        # print(f'[The number of thebest_data]: {cnt},  [The length of results(on unique sample id)]: {len(results)}')
        total_items = len(thebest_data)
        if cnt_all % (total_items // 10) == 0:
            percentage = cnt_all / total_items * 100
            logging.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>> {percentage:.0f}% complete')
            logging.debug(sample)
        
        final_results = {}
        for sample_id, sample in results.items():
            # pprint(v)
            final_results[sample_id] = words_to_response(args.task, sample)
            # pprint(final_results[k])
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>'*3)
    
    logging.info(f'[The number of thebest_data(cnt_all)]: {cnt_all}')
    logging.info(f'[The number of the predicted "empty"]: {cnt_empty}')
    logging.info(f'[The number of the predicted "dot"]: {cnt_dot}')
    logging.info(f'[The length of final_results]: {len(final_results)}')
    write_jsonl(list(final_results.values()), output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thebest_file", default=None, type=str, required=True) 
    parser.add_argument("--frame_file", default=None, type=str, required=True)
    # parser.add_argument("--preprocess_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True) 
    parser.add_argument("--task", default='dialfact', type=str)
    
    args = parser.parse_args()

    thebest_data = read_json_data(args.thebest_file)
    if args.task == 'augwow':
        examples = read_augwow_examples(args.frame_file)
    elif args.task == 'dialfact':
        examples = read_dialfact_examples(args.frame_file)

    combine_predictions(args, examples, thebest_data, args.output_file)#, args.preprocess_file)

if __name__ == "__main__":
    main()