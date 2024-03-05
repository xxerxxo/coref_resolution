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

nlp = spacy.load("en_core_web_sm")

# 모델과 토크나이저 초기화
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

def write_json(data, file_path):
    """
    Write data to a JSON file. w/ args.output_file
    """
    # logging.info(f'Writing {len(data)} items to {file_path}.')
    print(f'Writing {len(data)} items to {file_path}.')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    
def evaluate_sentence_with_candidates(sentence_template, candidates):
    # 모델과 토크나이저 초기화
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # model.eval()

    # 로그 확률을 계산하는 함수
    def score(sentence):
        tokenize_input = tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input])
        with torch.no_grad():
            loss = model(tensor_input, labels=tensor_input)[0]
        return -loss.item()

    # 각 후보에 대한 문장의 확률 계산
    scores = {}
    for candidate in candidates:
        candidate_sentence = sentence_template.format(candidate)
        candidate_score = score(candidate_sentence)
        scores[candidate] = candidate_score

    # 확률이 높은 순서대로 후보 정렬 및 출력
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_candidates

# def generate_filled_sentences(found_pronoun, context, response, pronoun_index, candidates):
#     # 문맥에서 마지막 두 문장을 가져옵니다.
#     context_part = " ".join(context[-2:])
    
#     # 대응하는 응답에서 대명사를 {}로 대체합니다.
#     doc = nlp(response)
#     if doc[pronoun_index] == found_pronoun: # 지시대명사를 빈칸으로 대체하기
#         doc[pronoun_index] = '{}'
#     response_with_placeholder = " ".join(doc)
    
#     # 완성된 문장 템플릿을 생성합니다.
#     sentence_template = f"{context_part} {response_with_placeholder}"
    
#     # 각 후보에 대한 문장의 확률을 계산하고 정렬합니다.
#     sorted_candidates = evaluate_sentence_with_candidates(sentence_template, candidates)
    
#     return sorted_candidates

def generate_filled_sentences(found_pronoun, context, response, pronoun_index, candidates):
    # 문맥에서 마지막 두 문장을 가져옵니다.
    context_part = " ".join(context[-2:])
    
    # 대응하는 응답을 토큰화합니다.
    doc = nlp(response)
    tokens = [token.text for token in doc]  # spacy 토큰을 문자열 리스트로 변환
    
    # 지시대명사의 인덱스에 해당하는 토큰을 '{}'로 대체합니다.
    if tokens[pronoun_index] == found_pronoun:
        tokens[pronoun_index] = '{}'
    response_with_placeholder = " ".join(tokens)  # 수정된 토큰 리스트를 다시 문자열로 결합
    
    # 완성된 문장 템플릿을 생성합니다.
    sentence_template = f"{context_part} {response_with_placeholder}"
    
    # 각 후보에 대한 문장의 확률을 계산하고 정렬합니다.
    sorted_candidates = evaluate_sentence_with_candidates(sentence_template, candidates)
    
    return sorted_candidates

def select_predictions(frame_file, pred_file, predictions_file, output_file):
    # logging.info('>>>>>>>>>>>>>>>>> Start selecting predictions.')
    print('>>>>>>>>>>>>>>>>> Start selecting predictions.')
    
    frames = []
    nbest_predictions = [] # several candidates with "TOP priority" with lowest log loss
    predictions = [] # only one "SECOND priority"
    with jsonlines.open(frame_file) as reader:
        for line in reader:
            frames.append(line)
    with open(pred_file, 'r') as f:
        nbest_predictions = json.load(f)
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # assert len(frames) == len(nbest_predictions)
    # logging.info('>>>>>>>>>>>>>>>>> len(frames): {}'.format(len(frames)))
    # logging.info('>>>>>>>>>>>>>>>>> len(predictions): {}'.format(len(predictions)))
    # logging.info('>>>>>>>>>>>>>>>>> len(nbest_predictions): {}'.format(len(nbest_predictions)))
    print('>>>>>>>>>>>>>>>>> len(frames): {}'.format(len(frames)))
    print('>>>>>>>>>>>>>>>>> len(predictions): {}'.format(len(predictions)))
    print('>>>>>>>>>>>>>>>>> len(nbest_predictions): {}'.format(len(nbest_predictions)))

    results = {}
    print()
    
    qasid_empty = []
    for i, frame in enumerate(frames):
        if str(i) in nbest_predictions:
            predicted_noun = "" #initialization
            candidates = [cand["text"] for cand in nbest_predictions[str(i)]]
            sorted_candidates = generate_filled_sentences(
                                                            frame['found_pronoun'],
                                                            frame["context_text"],
                                                            frame["orig_response"],
                                                            frame["pronoun_index"],
                                                            candidates
                                                        )   
            
            if not sorted_candidates: # No predictions: "empty" in predictsion_.json, [] in nbest_predictions_.json
                # logging.info(f'[Empty] ---> {frame["qas_id"]}')
                qasid_empty.append(frame['qas_id'])
                predicted_noun = predictions[frame['qas_id']]
                results[frame['qas_id']] = [predicted_noun]
            else:
                predicted_noun = sorted_candidates[0][0]
                # logging.info(f'[{predicted_noun}] ---> {frame["qas_id"]}')
                results[frame['qas_id']] = [predicted_noun]
                # logging.info(f'*****************************************************************************************')
            # break
        
            
        else: #NO predictions in nbest_predictions.json
            predicted_noun = predictions[frame['qas_id']]
            results[frame['qas_id']] = [predicted_noun]
           
        print(f'[{i}  sample] {frame["qas_id"]} : {predicted_noun}') 
        if i % (len(frames) // 10) == 0:
            print()
            # logging.info(f'>>>>>>>>>>>>>>>>> {i} / {len(frames)} done.')
            print(f'>>>>>>>>>>>>>>>>> {i} / {len(frames)} done.')
            print()
        
    pprint(results)
    write_json(results, output_file)
    # logging.info(f'>>>>>>>>>>>>>>>>> The Number of empty predictions: {len(qasid_empty)} out of {len(frames)}.')
    print(f'>>>>>>>>>>>>>>>>> The Number of empty predictions: {len(qasid_empty)} out of {len(frames)}.')

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