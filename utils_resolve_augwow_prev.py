import spacy
from collections import Counter, namedtuple
import json
import argparse
from pprint import pprint
import logging

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Spacy NLP initialization
nlp = spacy.load("en_core_web_sm")

# SquadExample Struct Definition
SquadExample = namedtuple("SquadExample", ["qas_id", "question_text", "doc_tokens", "is_impossible", "orig_answer_text", "start_position", "end_position", "found_pronoun", "orig_response", 'new_response']) #"idx_pronoun", 


def identify_pronouns(sentence):
    doc = nlp(sentence)
    pronouns = ['he', 'she', 'it', 'they', 'them', 'this', 'these', 'those', 'who', 'whom', 'which', 'whose']
    found_pronoun = ''
    for token in doc:
        if token.pos_ == 'PRON' and pronouns.__contains__(token.text):
            found_pronoun = token.text
        return found_pronoun


def identify_pronouns2(sentence):
    sentence = sentence.split()
    # print(f'sentence: {sentence}')
    pronouns = ['he', 'she', 'it', 'they', 'them', 'this', 'these', 'those', 'who', 'whom', 'which', 'whose']
    found_pronoun = ''
    for token in sentence:
        # print(f'token: {token}')
        if token.lower() in pronouns:
            found_pronoun = token
            return found_pronoun
        
def construct_question_text(pronoun, response):
    """Construct question text based on the identified pronoun."""
    if pronoun in ['he', 'she']:
        return f"In the context, '{response}', who does '{pronoun}' specifically refer to?"
    elif pronoun == 'it':
        return f"In the sentence, '{response}', what does '{pronoun}' refer to?"
    elif pronoun in ['they', 'them']:
        return f"In the narrative, '{response}', who or what are referred to as '{pronoun}'?"
    elif pronoun in ['this', 'that', 'these', 'those']:
        return f"In the discussion, '{response}', what specific item or situation does '{pronoun}' point to?"
    else:
        return f"Considering the context, '{response}', how is '{pronoun}' utilized or defined?"


def read_augwow_examples(input_file):
    logging.info(f'{input_file} : Augwow file loading...')
    
    examples = []
    samples_w_pronouns = []
    found_pronouns = []
    with open(input_file, "r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader):
            item = json.loads(line)  
            
            if idx % 1000 == 0:
                logging.info(f'AugWoW Loaded... {idx}th item processing...')
                
            response = item['claim'].split('[RESPONSE]: ')[-1]
            context = ' '.join(item['context'])
            
            found_pronoun = identify_pronouns2(response)
            if not found_pronoun:
                continue
            else:
                found_pronouns.append(found_pronoun)
                unique_id = f"{item['id']}-{idx}"
                samples_w_pronouns.append(unique_id)
                question_text = construct_question_text(found_pronoun, response)

                # print(f"pronouns: {found_pronoun}")
                # print(f"response: {response}")
                # print(f"context: {context}")
                # print(f"item_id: {item['id']}")
                # print(f"question_text: {question_text}")
                # print()

                example = SquadExample(
                                            found_pronoun=found_pronoun,
                                            orig_response=response,
                                            qas_id=unique_id,
                                            question_text=question_text,
                                            doc_tokens=context.split(),  
                                            is_impossible=False,  # 해당 질문에 대한 답변이 문서 내에 존재한다고 가정. 추론 시에는 대부분의 경우 이 값을 False로 설정. 
                                            orig_answer_text="",  # 추론 과정에서는 직접 사용되지 않음
                                            start_position=-1,  # 추론 시에는 사용되지 않음
                                            end_position=-1  # 추론 시에는 사용되지 않음
                                        )

                examples.append(example)
        logging.info(f'AugWoW Loaded... {len(samples_w_pronouns)} samples with pronouns found...')
        counter = Counter(found_pronouns)
        logging.info(counter)
        lowercase_counter = Counter({key.lower(): 0 for key in counter})
        for key, value in counter.items():
            lowercase_counter[key.lower()] += value
        logging.info(lowercase_counter)
        # write_jsonl(examples, 'augwow_examples.jsonl')
    return examples

def read_coref_file(coref_file):
    coref_data = []
    with open(coref_file, "r", encoding="utf-8") as reader:
        coref_data = json.load(reader)
    logging.info(f'{coref_file} : Coref_prediction file loaded...  : {len(coref_data)} items in coref_data...')
          
    return coref_data

def resolve_coref(examples, coref_data):
    logging.info(f'Resolving Coreference Problems in AugWow...')
    resolve_examples = []
    for unique_id, predicted_value in coref_data.items():
        for example in examples: # examples: AugWoW examples
            if example.qas_id == unique_id:
                response = example.orig_response
                found_pronoun = example.found_pronoun
                # idx_pronoun = example['idx_pronoun']
                response_new = response.replace(found_pronoun, predicted_value)
                logging.info(f'The found_pronoun: [{found_pronoun}] is replaced with [{predicted_value}]. -----> Response_new: {response_new}')
                example.new_response = response_new
                resolve_examples.append(example)
    
    logging.info(f'{len(resolve_examples)} claims were resolved in resolve_coref() function...')
    return resolve_examples

def write_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--train_file", default=None, type=str, required=False,
    #                     help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--coref_file", default=None, type=str, required=True) #json file 
    parser.add_argument("--output_file", default=None, type=str, required=True)

    args = parser.parse_args()

    coref_data = read_coref_file(args.coref_file)
    examples = read_augwow_examples(args.input_file)
    resolved = resolve_coref(examples, coref_data)
    write_jsonl(resolved, args.output_file)
    
    logging.info(f'len(coref_data): {len(coref_data)}, type(coref_data): {type(coref_data)}')
    logging.info(f'len(examples): {len(examples)}, type(examples): {type(examples)}')
    logging.info(f'len(resolved): {len(resolved)}, type(resolved): {type(resolved)}')
    

if __name__ == "__main__":
    main()
