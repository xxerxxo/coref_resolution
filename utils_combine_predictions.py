import logging
logging.basicConfig(level=logging.ERROR)

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
    Read coreference data from a file.
    --coref_file /data/scratch/acw722/corefbert/result/inference/predictions_v2.json
    """
    json_data = []
    with open(json_file_path, "r", encoding="utf-8") as reader:
        json_data = json.load(reader)
    print(f'Loaded coreference data for {len(json_data)} examples.')
    return json_data

def read_augwow_examples(input_file):
    """
    Read AugWoW examples with the found_pronoun.
    inpuf_file: /data/scratch/acw722/corefbert/result/resolved_data//augwow_with_pronouns.jsonl
    """
    examples = []
    with open(input_file, "r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader):
            item = json.loads(line)
            examples.append(item)
    print(f'Loaded {len(examples)} examples with pronouns found.')
    return examples

def read_dialfact_examples(input_file):
    """
    Read Dialfact examples with the found_pronoun.
    /data/scratch/acw722/corefbert/result/resolved_data/dialfact_valid_with_pronouns.jsonl\
    /data/scratch/acw722/corefbert/result/resolved_data/dialfact_test_with_pronouns.jsonl\
    """
    examples = []
    with open(input_file, "r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader):
            item = json.loads(line)
            examples.append(item)
    print(f'Loaded {len(examples)} examples with pronouns found.')
    return examples

def write_jsonl(data, file_path):
    print(f'Writing {len(data)} items to {file_path}.')
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def find_sample(examples, qas_id):
    for s in examples:
        if s['qas_id'] == qas_id:
            return s['item']
        
def words_to_response(sample):
    words = sample['words']
    new_sample = sample.deepcopy()
    new_sample['ori_response'] = new_sample['response']
    new_sample['response'] = ' '.join(words)
    return new_sample

def combine_predictions(args, examples, thebest_data, output_file, preprocess_file):
    results = {} # key: qas_id, value: item

    if args.task == 'dialfact':
        cnt = 0
        for qas_id, pred_noun in thebest_data.items():

            sample_id = qas_id[:qas_id[:qas_id.rfind('_')].rfind('_')]
            pronoun_idx = qas_id[qas_id.rfind('_')+1:]
            # print(qas_id, '--->', sample_id)
            # print(qas_id, '--->', pronoun_idx)
            if sample_id in results: # 이미 results안에 sample이 있는 경우
                sample = results[sample_id]
                sample['words'][int(pronoun_idx)] = pred_noun[0]
            else: # results안에 sample이 없는 경우, 처음으로 등장하는 sample인 경우
                sample = find_sample(examples, qas_id)
                dict_words = {i: item for i, item in enumerate([token.text for token in nlp(sample['response'])])}
                dict_words[int(pronoun_idx)] = pred_noun[0]
                sample['words'] = dict_words
                results[sample_id] = sample
            
            cnt += 1
            total_items = len(thebest_data)
            if cnt % (total_items // 10) == 0:
                percentage = cnt / total_items * 100
                print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>> {percentage:.0f}% complete')
                pprint(sample)
            
    elif args.task == 'augwow':
        cnt = 0
        for qas_id, pred_noun in thebest_data.items():

            sample_id = qas_id[:qas_id[:qas_id.rfind('_')].rfind('_')]
            pronoun_idx = qas_id[qas_id.rfind('_')+1:]
            # print(qas_id, '--->', sample_id)
            # print(qas_id, '--->', pronoun_idx)
            if sample_id in results: # 이미 results안에 sample이 있는 경우
                sample = results[sample_id]
                sample['words'][int(pronoun_idx)] = pred_noun[0]
            else: # results안에 sample이 없는 경우, 처음으로 등장하는 sample인 경우
                sample = find_sample(examples, qas_id)
                ori_response = sample['claim'].split('[REPSONSE]: ')[-1]
                dict_words = {i: item for i, item in enumerate([token.text for token in nlp(ori_response)])}
                dict_words[int(pronoun_idx)] = pred_noun[0]
                sample['words'] = dict_words
                results[sample_id] = sample
            
            cnt += 1
            total_items = len(thebest_data)
            if cnt % (total_items // 10) == 0:
                percentage = cnt / total_items * 100
                print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>> {percentage:.0f}% complete')
                pprint(sample)


    write_jsonl(list(results.values()), output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thebest_file", default=None, type=str, required=True) 
    parser.add_argument("--frame_file", default=None, type=str, required=True)
    parser.add_argument("--preprocess_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True) 
    parser.add_argument("--task", default=128, type=str)
    parser.add_argument("--prefix", default=128, type=str)
    
    args = parser.parse_args()

    thebest_data = read_json_data(args.thebest_file)
    if args.task == 'augwow':
        examples = read_augwow_examples(args.frame_file)
    elif args.task == 'dialfact':
        examples = read_dialfact_examples(args.frame_file)

    combine_predictions(args, examples, thebest_data, args.output_file, args.preprocess_file)

if __name__ == "__main__":
    main()