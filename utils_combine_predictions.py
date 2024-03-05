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

def read_coref_data(coref_file):
    """
    Read coreference data from a file.
    --coref_file /data/scratch/acw722/corefbert/result/inference/predictions_v2.json
    """
    coref_data = []
    with open(coref_file, "r", encoding="utf-8") as reader:
        coref_data = json.load(reader)
    logging.info(f'Loaded coreference data for {len(coref_data)} examples. Type of coref_data: {type(coref_data)}')
    return coref_data

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
    logging.info(f'Loaded {len(examples)} examples with pronouns found.')
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
    logging.info(f'Loaded {len(examples)} examples with pronouns found.')
    return examples

def write_jsonl(data, file_path):
    """
    Write data to a JSONL file.
    """
    logging.info(f'Writing {len(data)} items to {file_path}.')
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def combine_predictions(args, examples, nbest_coref_data, output_file, preprocess_file):
    pass

def main():
    parser = argparse.ArgumentParser()
    #/data/scratch/acw722/corefbert/result/inference/thebest_predictions_dialfact_valid.json
    parser.add_argument("--pred_file", default=None, type=str, required=True) 
    #/data/scratch/acw722/corefbert/result/resolved_data/dialfact/dialfact_valid_with_pronouns.jsonl
    parser.add_argument("--frame_file", default=None, type=str, required=True)
    #/data/scratch/acw722/corefbert/result/resolved_data/dialfact/preprocess_valid_w_pronouns.jsonl
    parser.add_argument("--preprocess_file", default=None, type=str, required=True)
    #/data/scratch/acw722/corefbert/result/resolved_data/dialfact/valid_w_pronouns.jsonl
    parser.add_argument("--output_file", default=None, type=str, required=True) 
    
    args = parser.parse_args()

    nbest_coref_data = read_coref_data(args.pred_file)
    if args.task == 'augwow':
        examples = read_augwow_examples(args.frame_file)
    elif args.task == 'dialfact':
        examples = read_dialfact_examples(args.frame_file)

    combine_predictions(args, examples, nbest_coref_data, args.output_file, args.preprocess_file)

if __name__ == "__main__":
    main()