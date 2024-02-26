from pprint import pprint
import spacy
import json
import logging
from collections import Counter
import argparse

# Spacy NLP initialization
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# def replace_pronoun_with_noun(response, predicted_noun, found_pronoun, pronoun_index):
#     """
#     Replace the pronoun with the predicted noun in the response.
#     """
#     logging.info(f'Replacing {found_pronoun} with {predicted_noun}(idx: {pronoun_index}) in response: {response}')
#     # Tokenize the response
#     response_tokens = response.split()
#     new_response = None
#     if response_tokens[pronoun_index] != found_pronoun:
        
#         logging.error(f'Pronoun mismatch: {response_tokens[pronoun_index]} != {found_pronoun}')
#         doc = nlp(response)
#         for i, token in enumerate(doc):
#             if token.text.lower() == found_pronoun.lower():
#                 logging.info(f'Pronoun found at index {i}: {token.text} in {doc}. And predicted_noun: {predicted_noun}')
#                 new_response = doc[:i].text + predicted_noun + doc[i+1:].text
#                 logging.info(f'new_response: {new_response}')
#                 return new_response
    
#     # Ensure predicted_noun is a string
#     if isinstance(predicted_noun, list):
#         if len(predicted_noun) > 0:
#             predicted_noun = predicted_noun[0]  # Use the first item if it's a list
#         else:
#             logging.error('Predicted noun list is empty.')
#             return new_response
    
#     logging.info(f'response_tokens: {response_tokens}')
#     logging.info(f'predicted_noun: {predicted_noun}')
#     logging.info(f'found_pronoun: {found_pronoun}')
#     logging.info(f'pronoun_index: {pronoun_index}')
#     # Replace the pronoun with the predicted noun
#     response_tokens[pronoun_index] = predicted_noun
#     # Join the tokens to form the new response
#     new_response = ' '.join(response_tokens)
#     logging.info(f'new_response: {new_response}')
#     return new_response

def replace_pronoun_with_noun(response, predicted_noun, found_pronoun, pronoun_index, item_id):
    """
    Replace the pronoun with the predicted noun in the response.
    """
    # Ensure predicted_noun is a string
    if isinstance(predicted_noun, list) and len(predicted_noun) > 0:
        predicted_noun = predicted_noun[0]  # Use the first item if it's a list
    elif isinstance(predicted_noun, list) and len(predicted_noun) == 0:
        logging.error('Predicted noun list is empty. [item_id: {item_id}]')
        return response

    # Tokenize the response
    doc = nlp(response)
    if pronoun_index >= len(doc) or doc[pronoun_index].text.lower() != found_pronoun.lower():
        logging.error(f'Pronoun mismatch or index out of range. [item_id: {item_id}]')
        return response

    # logging.info(f'Replacing {found_pronoun} with {predicted_noun} at index {pronoun_index} in response: {response}')

    # Replace the pronoun with the predicted noun
    response_tokens = [token.text_with_ws for token in doc]  # Preserve whitespace
    if response_tokens[pronoun_index][0].isupper(): # Capitalize the predicted noun if the pronoun is capitalized
        predicted_noun = predicted_noun.capitalize()
        response_tokens[pronoun_index] = predicted_noun + doc[pronoun_index].whitespace_
    new_response = ''.join(response_tokens)
    # logging.info(f'new_response: {new_response}')
    return new_response


def write_jsonl(data, file_path):
    """
    Write data to a JSONL file.
    """
    logging.info(f'Writing {len(data)} items to {file_path}.')
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def resolve_coref_with_examples(examples, coref_data, output_file, preprocess_file):
    """
    Resolve coreference problems in examples using the provided coreference data.
    """
    logging.info(f'Resolving coreference in {len(examples)} examples using coreference data for {len(coref_data)} examples.')
    logging.info(f'Type of examples: {type(examples)}, and type of coref_data: {type(coref_data)}')

    # Read samples in examples and check id with the key in coref_data, if match, replace the pronoun with the value of the coref_data.
    # But, if the value in the coref_data is 'empty', then keep the original pronoun.
    # Count the number of samples with resolved pronouns when processing above.
    resolved_examples = [] # Resolved examples : all information from examples and new_response
    resolved_samples = [] # Resolved samples : only new_response and the original examples
    resolved_count = 0
    '''one example in /data/scratch/acw722/corefbert/result/resolved_data/augwow_with_pronouns.jsonl
    
    qas_id: "1845___2_antadj_195168"
    question_text: "Considering the context, 'League is a multiplayer off-line battle arena game , made by Riot games . It 's one of the top off-line games in the world at the moment !', how is 'It' utilized or defined?"
    doc_tokens: ["I", "love", "the", "game", "League", "of", "Legends!", "Have", "you", "ever", "heard", "of", "it", "or", "played", "it", "before?", "I", "think", "I've", "heard", "the", "name", "but", "I", "know", "nothing", "about", "it."]
    is_impossible: false
    orig_answer_text: ""
    start_position: -1
    end_position: -1
    found_pronoun: "It"
    orig_response: "League is a multiplayer off-line battle arena game , made by Riot games . It 's one of the top off-line games in the world at the moment !"
    new_response: ""
    context_text: ["I love the game League of Legends! Have you ever heard of it or played it before?", "I think I've heard the name but I know nothing about it."]
    pronoun_index: 16
    predicted_pronoun: null
    item: 
        id: "1845___2_antadj"
        evidence: [["League of Legends", 0, "League of Legends (abbreviated LoL) is a multiplayer online battle arena video game developed and published by Riot Games for Microsoft Windows and macOS.", 1]]
        claim: "[CONTEXT]: I love the game League of Legends! Have you ever heard of it or played it before? [EOT] I think I've heard the name but I know nothing about it. [RESPONSE]: League is a multiplayer off-line battle arena game , made by Riot games . It 's one of the top off-line games in the world at the moment !"
        context: ["I love the game League of Legends! Have you ever heard of it or played it before?", "I think I've heard the name but I know nothing about it."]
        label: "REFUTES"
        original_claim: "League is a multiplayer online battle arena game , made by Riot games . It 's one of the top online games in the world at the moment !"

    '''
    # augwow = augwow[:100]
    for idx, sample in enumerate(examples):
        # logging.info(f'[{idx}]'+'*'*50)
        item_id = sample['qas_id']
        predicted_noun = coref_data.get(item_id, None)[0] # Coreference Noun, if it's 'empty', then keep the original pronoun
        # print(f'predicted_noun: {predicted_noun}')
        if sample['pronoun_index'] != -1 and predicted_noun != 'empty': # If the item_id is in coref_data, replace the pronoun with the predicted noun
            resolved_count += 1
            pronoun_index = sample['pronoun_index'] # Pronoun index in the original response
            found_pronoun = sample['found_pronoun'] # Pronoun in the original response
            response = sample['orig_response']
            new_response = replace_pronoun_with_noun(response, predicted_noun, found_pronoun, pronoun_index, item_id)

            if new_response:
                sample['new_response'] = new_response
                sample['predicted_pronoun'] = predicted_noun
                
                ### Change the response with coreference resolved
                # augwow
                # sample['item']['claim'] = sample['item']['claim'].split('[RESPONSE]:')[0] + f'[RESPONSE]: {new_response}'
                # dialfact
                sample['item']['response'] = new_response

                # Add coreference info in the augwow
                sample['item']['coref_noun'] = predicted_noun
                sample['item']['pronoun_idx'] = sample['pronoun_index']
                sample['item']['found_pronoun'] = sample['found_pronoun']
                sample['item']['qas_id'] = item_id
                sample['item']['question_text'] = sample['question_text']
            
            elif not new_response:
                logging.error(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Failed to replace pronoun with noun in example {idx}.')
                logging.error(f'Example: {sample}')
                logging.error(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Failed to replace pronoun with noun in example {idx}.')
                
            if idx%10000 == 0:
                logging.info(f'*'*50+'Processed {idx} examples.')
                logging.info(f'Example: {sample}')
                logging.info('*'*50)
            
        resolved_examples.append(sample)
        resolved_samples.append(sample['item'])

    logging.info(f'Resolved {resolved_count} examples.')
    
    write_jsonl(resolved_examples, preprocess_file)
    logging.info(f'Wrote resolved examples to {preprocess_file}.')
    
    # OUTPUT FILE
    write_jsonl(resolved_samples, output_file)
    logging.info(f'Wrote resolved samples to be trained rightly to {output_file}.')
    

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--coref_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True) # 공동참조 해결 후, 바로 훈련에 들어가도되는 데이터
    parser.add_argument("--preprocess_file", default=None, type=str, required=True) # 공동참조 해결 관련 모든 정보
    parser.add_argument("--task", default=128, type=str)
    parser.add_argument("--prefix", default=128, type=str)

    args = parser.parse_args()

    coref_data = read_coref_data(args.coref_file)
    if args.task == 'augwow':
        examples = read_augwow_examples(args.input_file)
    elif args.task == 'dialfact':
        examples = read_dialfact_examples(args.input_file)

    #save resolved examples to output_file
    resolve_coref_with_examples(examples, coref_data, args.output_file, args.preprocess_file)

if __name__ == "__main__":
    main()
