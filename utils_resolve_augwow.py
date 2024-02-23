import spacy
import json
import logging
from collections import Counter
import argparse

# Spacy NLP initialization
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def replace_pronoun_with_noun(response, predicted_noun, found_pronoun, pronoun_index):
    """
    Replace the pronoun with the predicted noun in the response.
    """
    # Tokenize the response
    response_tokens = response.split()
    
    if response_tokens[pronoun_index] != found_pronoun:
        logging.error(f'Pronoun mismatch: {response_tokens[pronoun_index]} != {found_pronoun}')
        return response
    
    # Replace the pronoun with the predicted noun
    response_tokens[pronoun_index] = predicted_noun
    # Join the tokens to form the new response
    new_response = ' '.join(response_tokens)
    return new_response


def resolve_coref_with_augwow(augwow, coref_data, output_file):
    """
    Resolve coreference problems in AugWoW examples using the provided coreference data.
    """
    logging.info(f'Resolving coreference in {len(augwow)} examples using coreference data for {len(coref_data)} examples.')
    logging.info(f'Type of augwow: {type(augwow)}, and type of coref_data: {type(coref_data)}')

    # Read samples in augwow and check id with the key in coref_data, if match, replace the pronoun with the value of the coref_data.
    # But, if the value in the coref_data is 'empty', then keep the original pronoun.
    # Count the number of samples with resolved pronouns when processing above.
    resolved_examples = []
    resolved_count = 0
    '''
    - one example in augwow:
    {"qas_id": "1845___2_antadj_195168", 
    "question_text": "Considering the context, 'League is a multiplayer off-line battle arena game , made by Riot games . It 's one of the top off-line games in the world at the moment !', how is 'It' utilized or defined?", 
    "doc_tokens": ["I", "love", "the", "game", "League", "of", "Legends!", "Have", "you", "ever", "heard", "of", "it", "or", "played", "it", "before?", "I", "think", "I've", "heard", "the", "name", "but", "I", "know", "nothing", "about", "it."], 
    "is_impossible": false, "orig_answer_text": "", "start_position": -1, "end_position": -1, 
    "found_pronoun": "It", "orig_response": "", "new_response": null, 
    "context_text": ["I love the game League of Legends! Have you ever heard of it or played it before?", "I think I've heard the name but I know nothing about it."], 
    "pronoun_index": 16, "predicted_pronoun": null, "item": {}}
    '''
    for idx, item in enumerate(augwow):
        item_id = item['qas_id']
        predicted_noun = coref_data.get(item_id, None) # Coreference Noun
        if predicted_noun and item['pronoun_index'] != -1: # If the item_id is in coref_data, replace the pronoun with the predicted noun
            resolved_count += 1
            pronoun_index = item['pronoun_index'] # Pronoun index in the original response
            found_pronoun = item['found_pronoun'] # Pronoun in the original response
            response = item['orig_response']
            new_response = replace_pronoun_with_noun(response, predicted_noun, found_pronoun, pronoun_index)
            item['new_response'] = new_response
            resolved_examples.append(item)
        else: # If the item_id is not in coref_data, keep the original pronoun 
            resolved_examples.append(item)
    logging.info(f'Resolved {resolved_count} examples.')
    logging.info(f'Wrote resolved examples to {output_file}.')
    write_jsonl(resolved_examples, output_file)
    

def write_jsonl(data, file_path):
    """
    Write data to a JSONL file.
    """
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--coref_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)

    args = parser.parse_args()

    coref_data = read_coref_data(args.coref_file)
    examples = read_augwow_examples(args.input_file)

    #save resolved examples to output_file
    resolve_coref_with_augwow(examples, coref_data, args.output_file)

if __name__ == "__main__":
    main()
