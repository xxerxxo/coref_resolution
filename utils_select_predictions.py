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

# 모델과 토크나이저 초기화
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

def write_json(data, file_path):
    """
    Write data to a JSON file. w/ args.output_file
    """
    logging.info(f'Writing {len(data)} items to {file_path}.')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    
def evaluate_sentence_with_candidates(sentence_template, candidates):
    # 모델과 토크나이저 초기화
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

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

def generate_filled_sentences(context, response, pronoun_index, candidates):
    # 문맥에서 마지막 두 문장을 가져옵니다.
    context_part = " ".join(context[-2:])
    
    # 대응하는 응답에서 대명사를 {}로 대체합니다.
    response_tokens = response.split()
    response_tokens[pronoun_index] = '{}'
    response_with_placeholder = " ".join(response_tokens)
    
    # 완성된 문장 템플릿을 생성합니다.
    sentence_template = f"{context_part} {response_with_placeholder}"
    
    # 각 후보에 대한 문장의 확률을 계산하고 정렬합니다.
    sorted_candidates = evaluate_sentence_with_candidates(sentence_template, candidates)
    
    return sorted_candidates

def select_predictions(frame_file, pred_file, output_file):
    frames = []
    predictions = []
    with jsonlines.open(frame_file) as reader:
        for line in reader:
            frames.append(line)
    with open(pred_file, 'r') as f:
        predictions = json.load(f)

    assert len(frames) == len(predictions)
    results = {}

    for i, frame in enumerate(frames):
        candidates = [cand["text"] for cand in predictions[str(i)]]
        sorted_candidates = generate_filled_sentences(
                                                        frame["context_text"],
                                                        frame["orig_response"],
                                                        frame["pronoun_index"],
                                                        candidates
                                                    )   
        results[frame['qas_id']] = [sorted_candidates[0][0]]
    pprint(results)
    # write_json(results, output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_file", default=None, type=str, required=True) #jsonl
    parser.add_argument("--pred_file", default=None, type=str, required=True) #json
    parser.add_argument("--output_file", default=None, type=str, required=True) #json
    
    args = parser.parse_args()

    select_predictions(args.frame_file, args.pred_file, args.output_file)

if __name__ == "__main__":
    main()