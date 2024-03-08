import logging
from pprint import pprint
import spacy
import json, jsonlines
from collections import Counter
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 설정된 로깅 레벨을 INFO로 변경하여 더 많은 정보를 로그에 포함시킵니다.
logging.basicConfig(level=logging.INFO)

nlp = spacy.load("en_core_web_sm")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

def write_json(data, file_path):
    logging.info(f'Writing {len(data)} items to {file_path}.')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def evaluate_sentence_with_candidates(sentence_template, candidates):
    def score(sentence):
        tokenize_input = tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input])
        with torch.no_grad():
            loss = model(tensor_input, labels=tensor_input)[0]
        return -loss.item()

    scores = {}
    for candidate in candidates:
        candidate_sentence = sentence_template.format(candidate)
        candidate_score = score(candidate_sentence)
        scores[candidate] = candidate_score

    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_candidates

def generate_filled_sentences(found_pronoun, context, response, pronoun_index, candidates):
    context_part = " ".join(context[-2:])
    doc = nlp(response)
    tokens = [token.text for token in doc]
    
    if tokens[pronoun_index] == found_pronoun:
        tokens[pronoun_index] = '{}'
    response_with_placeholder = " ".join(tokens)
    
    sentence_template = f"{context_part} {response_with_placeholder}"
    
    sorted_candidates = evaluate_sentence_with_candidates(sentence_template, candidates)
    
    return sorted_candidates

def select_predictions(frame_file, pred_file, predictions_file, output_file):
    logging.info('Start selecting predictions.')
    
    frames = []
    nbest_predictions = []
    predictions = []
    with jsonlines.open(frame_file) as reader:
        for line in reader:
            frames.append(line)
    with open(pred_file, 'r') as f:
        nbest_predictions = json.load(f)
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    logging.info(f'len(frames): {len(frames)}')
    logging.info(f'len(predictions): {len(predictions)}')
    logging.info(f'len(nbest_predictions): {len(nbest_predictions)}')

    results = {}
    
    qasid_empty = []

    for i, frame in enumerate(frames):
        if str(i) in nbest_predictions:
            predicted_noun = ""
            
            candidates = [cand["text"] for cand in nbest_predictions[str(i)]]
            
            sorted_candidates = generate_filled_sentences(
                                                            frame['found_pronoun'],
                                                            frame["context_text"],
                                                            frame["orig_response"],
                                                            frame["pronoun_index"],
                                                            candidates
                                                        )
            if not sorted_candidates:
                qasid_empty.append(frame['qas_id'])
                predicted_noun = predictions[frame['qas_id']]
                results[frame['qas_id']] = [predicted_noun]
            else:
                predicted_noun = sorted_candidates[0][0]
                results[frame['qas_id']] = [predicted_noun]

        else:
            predicted_noun = predictions[frame['qas_id']]
            results[frame['qas_id']] = [predicted_noun]
        
        logging.info(f'[{i} sample] {frame["qas_id"]} : {predicted_noun}') 
        if i % (len(frames) // 10) == 0:
            logging.info(f'{i} / {len(frames)} done.')
            logging.info(f'*****************************************************************************')
        
    # pprint(results)
    write_json(results, output_file)
    logging.info(f'The Number of empty predictions: {len(qasid_empty)} out of {len(frames)}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_file", default=None, type=str, required=True) #jsonl
    parser.add_argument("--pred_file", default=None, type=str, required=True) #json
    parser.add_argument("--output_file", default=None, type=str, required=True) #json
    parser.add_argument("--predictions_file", default=None, type=str, required=True) #json
    
    args = parser.parse_args()
    pprint(args)

    select_predictions(args.frame_file, args.pred_file,args.predictions_file, args.output_file)

if __name__ == "__main__":
    main()