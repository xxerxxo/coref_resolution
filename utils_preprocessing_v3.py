from pprint import pprint
import spacy
import logging
from collections import Counter
import argparse
import json

# Spacy NLP initialization
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define pronouns list globally
PRONOUNS = ['he', 'she', 'it', 'they', 'them', 'this', 'these', 'those', 'who', 'whom', 'which', 'whose']

class SquadExample:
    def __init__(self, qas_id, question_text, doc_tokens, is_impossible, orig_answer_text="", 
                 start_position=-1, end_position=-1, found_pronoun="", orig_response="", new_response="", 
                 context_text="", pronoun_index=-1, predicted_pronoun=None, item={},
                 answer_text = '', start_position_character = -1):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.is_impossible = is_impossible
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.found_pronoun = found_pronoun
        self.orig_response = orig_response
        self.new_response = new_response
        self.context_text = context_text
        self.pronoun_index = pronoun_index
        self.predicted_pronoun = predicted_pronoun

        # AugWoW 데이터셋의 원본 필드
        self.item = item

    def to_dict(self):
        """Converts the SquadExample instance into a dictionary."""
        return {
            "qas_id": self.qas_id,
            "question_text": self.question_text,
            "doc_tokens": self.doc_tokens,
            "is_impossible": self.is_impossible,
            "orig_answer_text": self.orig_answer_text,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "found_pronoun": self.found_pronoun,
            "orig_response": self.orig_response,
            "new_response": self.new_response,
            "context_text": self.context_text,
            "pronoun_index": self.pronoun_index,
            "predicted_pronoun": self.predicted_pronoun,
            "item": self.item
        }


# def construct_question_text(pronoun, response):
#     """Construct question text based on the identified pronoun."""
#     if pronoun in ['he', 'she']:
#         return f"In the context, '{response}', who does '{pronoun}' specifically refer to?"
#     elif pronoun == 'it':
#         return f"In the sentence, '{response}', what does '{pronoun}' refer to?"
#     elif pronoun in ['they', 'them']:
#         return f"In the narrative, '{response}', who or what are referred to as '{pronoun}'?"
#     elif pronoun in ['this', 'that', 'these', 'those']:
#         return f"In the discussion, '{response}', what specific item or situation does '{pronoun}' point to?"
#     else:
#         return f"Considering the context, '{response}', how is '{pronoun}' utilized or defined?"

def construct_query(pronoun, response):
    """Construct a QUOREF style question directly incorporating different types of pronouns."""
    intro = "Based on the context,"

    # `us` is considered in the group for "who or what does 'pronoun' refer to?"
    if pronoun.lower() in ['i', 'me', 'we', 'us', 'you']:
        return f"{intro} who or what does '{pronoun}' refer to in '{response}'?"

    elif pronoun.lower() in ['he', 'she', 'it']:
        return f"{intro} who does '{pronoun}' refer to in '{response}'?"

    # Including 'us' in the possessive pronouns group is not suitable due to its usage.
    elif pronoun.lower() in ['his', 'her', 'hers', 'its', 'my', 'our', 'ours', 'your', 'yours', 'their', 'theirs']:
        return f"{intro} whose '{pronoun}' is mentioned in '{response}'?"

    elif pronoun.lower() in ['they', 'them']:
        return f"{intro} who are referred to as '{pronoun}' in '{response}'?"
    
    else:
        return f"{intro} how is '{pronoun}' used in '{response}'?"


def construct_question_text(pronoun, response):
    """Construct question text based on the identified pronoun, more aligned with QUOREF style."""
    context_phrase = f"Considering the context, '{response}',"
    
    if pronoun.lower() in ['he', 'she']:
        action_or_description = "is mentioned as" if "mentioned" in response else "is described as"
        return f"{context_phrase} which person {action_or_description} '{pronoun}'?"
    elif pronoun.lower() == 'it':
        detail_hint = "referring to an object, concept, or situation"
        return f"{context_phrase} what exactly does '{pronoun}' {detail_hint}?"
    elif pronoun.lower() in ['they', 'them']:
        group_or_entity = "individuals or entities"
        return f"{context_phrase} which {group_or_entity} are collectively referred to as '{pronoun}'?"
    elif pronoun.lower() in ['this', 'that', 'these', 'those']:
        specificity_hint = "specific item, situation, or idea"
        return f"{context_phrase} what {specificity_hint} does '{pronoun}' point to?"
    else:
        general_use = "utilized or defined within the narrative"
        return f"{context_phrase} how is '{pronoun}' {general_use}?"


## Define pronouns list globally
# PRONOUNS = ['he', 'she', 'it', 'they', 'them', 'this', 'these', 'those', 'who', 'whom', 'which', 'whose']

def identify_pronouns(sentence):
    """
    Identify all pronouns in the response sentence.
    Returns a list of "a tuple of  its index and pronoun. If no pronouns are found, returns an empty list.
    """
    doc = nlp(sentence)
    result = []
    for token_idx, token in enumerate(doc):
        if token.text.lower() in PRONOUNS and token.pos_ == 'PRON':
            result.append((token_idx, token.text)) #(pronoun_index, pronoun_text)
    return result

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

"""
- tag: dialfact/augwow
- input_file: input file path
"""
def read_examples(input_file, tag='dialfact', type='all', cnt_ctx=1): 
    examples = []
    dict_examples = []

    ori_examples = read_jsonl(input_file) # filepath, 'dialfact' or 'augwow'
    ori_examples = ori_examples[:10] # for testing

    for sample_idx, sample in enumerate(ori_examples):
        
        # slicing context
        if cnt_ctx > len(sample['context']):  # sample['context']의 길이가 cnt_ctx보다 작은 경우를 처리
            sample['context'] = sample['context']  # 리스트 전체를 유지
        else:
            sample['context'] = sample['context'][-cnt_ctx:]  # 마지막 cnt_ctx 개의 요소를 슬라이싱
        
        ctx = ' '.join(sample['context'])
        doc_tokens = ctx.split()
        
        if tag=='dialfact':
            response = sample['response']
        elif tag=='augwow':
            response = sample['claim'].split('[RESPONSE]: ')[-1]
        pronoun_info = identify_pronouns(response) # [(pronoun_idx, pronoun_text), (pronoun_idx, pronoun_text), ...]
        samples_for_each_response = [] # list of SquadExample for each pronouns in a sample, 샘플 하나당 여러개의 SquadExample이 생성될 수 있음
        if pronoun_info:
            logging.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> pronoun_info: {pronoun_info}')
            for pronoun_index, pronoun_text in pronoun_info:
                question_text = construct_query(pronoun_text, response) # construct question text with the pronoun
                qas_id = sample['id'] + '_' + str(sample_idx) + '_' + str(pronoun_index) # Unique ID for each example for predicting the reference noun to each sample
                # cnt_samples_with_prounoun += 1
                squad_example = SquadExample(
                                                qas_id=qas_id,
                                                question_text=question_text,
                                                context_text=sample['context'],
                                                
                                                doc_tokens=doc_tokens,
                                                found_pronoun=pronoun_text,
                                                pronoun_index=pronoun_index,
                                                
                                                is_impossible=False,

                                                answer_text='', # Not used in inference
                                                start_position_character=-1, # Not used in inference

                                                item = sample, # Store the original item 
                                                orig_response=response
                                            )     
                samples_for_each_response.append(squad_example)
        examples.extend(samples_for_each_response) # used in run_squad.py
        dict_examples.extend(list(map(lambda s: s.to_dict(), samples_for_each_response))) # used for resolving later
    logging.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Loaded {len(examples)} examples.')
    return examples, dict_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--coref_file", default=None, type=str, required=False)
    parser.add_argument("--output_file", default=None, type=str, required=False)

    args = parser.parse_args()

    read_dialfact_examples_w_pronouns(args.input_file)

if __name__ == "__main__":
    main()
