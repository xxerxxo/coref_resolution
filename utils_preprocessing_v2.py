from pprint import pprint
import spacy
import json
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

def read_augwow_examples_w_pronouns(input_file):
    """
    Input: All samples
    Output: Samples w/ pronouns
    """
    examples = []
    dict_examples = []

    # [1] READ AugWoW EXAMPLES
    original_augwow = []
    with open(input_file, "r", encoding="utf-8") as reader:
        original_augwow = [json.loads(line) for idx, line in enumerate(reader)]

    # original_augwow = original_augwow[:10]

    # [2] Find pronouns in the response & [3] Construct question text & [4] Create SquadExample
    for sample_idx, sample in enumerate(original_augwow):
        # logging.info(f'[DIALFACT] INDEX OF SAMPLE: {sample_idx} ')
        ctx = ' '.join(sample['context'])
        doc_tokens = ctx.split()  # doc_tokens is a list of tokens
        response = sample['claim'].split('[RESPONSE]: ')[-1]
        pronoun_info = identify_pronouns(response) # [(pronoun_idx, pronoun_text), (pronoun_idx, pronoun_text), ...]
        samples_for_each_response = [] # list of SquadExample for each response
        # logging.info(f'[RESPONSE CLAIM]: {response}')
        # logging.info(f'[PRONOUN INFO]: {pronoun_info}')
        if pronoun_info:
            for pronoun_index, pronoun_text in pronoun_info: # idx of pronoun in the response, text of pronoun
                question_text = construct_question_text(pronoun_text, response) # construct question text with the pronoun
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
        # logging.info(f'SAMPLE - [{sample_idx}] ADDED---------> NOW: [{len(examples)}] samples in examples, [{len(dict_examples)}] samples in dict_examples')
    # logging.info('*'*50)
    logging.info(f'Loaded {len(examples)} examples.')
    # logging.info(f'The number of samples with pronoun: {cnt_samples_with_prounoun}, percentage: {cnt_samples_with_prounoun/len(examples)*100:.2f}%')
    return examples, dict_examples

def read_dialfact_examples_w_pronouns(input_file):
    """
    Input: All samples
    Output: Samples w/ pronouns
    """
    examples = []
    dict_examples = []
    
    cnt_samples_with_prounoun = 0
    
    # [1] READ DIALFACT EXAMPLES
    original_dialfact = []
    with open(input_file, "r", encoding="utf-8") as reader:
        original_dialfact = [json.loads(line) for idx, line in enumerate(reader)]

    # [2] Find pronouns in the response & [3] Construct question text & [4] Create SquadExample
    for sample_idx, sample in enumerate(original_dialfact):
        # logging.info(f'[DIALFACT] INDEX OF SAMPLE: {sample_idx} ')
        ctx = ' '.join(sample['context'])
        doc_tokens = ctx.split()  # doc_tokens is a list of tokens
        response = sample['response']
        pronoun_info = identify_pronouns(response) # [(pronoun_idx, pronoun_text), (pronoun_idx, pronoun_text), ...]
        samples_for_each_response = [] # list of SquadExample for each response
        # logging.info(f'[RESPONSE CLAIM]: {response}')
        # logging.info(f'[PRONOUN INFO]: {pronoun_info}')
        if pronoun_info:
            for pronoun_index, pronoun_text in pronoun_info: # idx of pronoun in the response, text of pronoun
                question_text = construct_question_text(pronoun_text, response) # construct question text with the pronoun
                qas_id = sample['id'] + '_' + str(sample_idx) + '_' + str(pronoun_index) # Unique ID for each example for predicting the reference noun to each sample
                cnt_samples_with_prounoun += 1
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
        
        """
        IGNORE SAMPLES WITHOUT PRONOUNS
        """
        # else: # no pronouns found in the response
            
        #     qas_id = sample['id'] + '_' + str(sample_idx) + '_0000'
        #     squad_example = SquadExample(
        #                                     qas_id=qas_id,
        #                                     question_text=None,
        #                                     context_text=sample['context'],
                                            
        #                                     doc_tokens=doc_tokens,
        #                                     found_pronoun='',
        #                                     pronoun_index=-1,
                                            
        #                                     is_impossible=True,

        #                                     answer_text='', # Not used in inference
        #                                     start_position_character=-1, # Not used in inference

        #                                     item = sample, # Store the original item 
        #                                     orig_response=response
        #                                 )               
        #     samples_for_each_response.append(squad_example)
        #     print(f'[NO PRONOUN] {qas_id}')    

        # if sample_idx % 1000 == 0: #augwow has 190k examples
        #     logging.info(f'Until now, loaded {len(examples)} examples.')
        #     logging.info(json.dumps(squad_example.to_dict())+'\n')

        # print all samples in samples_for_each_response as a dict
        examples.extend(samples_for_each_response) # used in run_squad.py
        dict_examples.extend(list(map(lambda s: s.to_dict(), samples_for_each_response))) # used for resolving later
        # logging.info(f'SAMPLE - [{sample_idx}] ADDED---------> NOW: [{len(examples)}] samples in examples, [{len(dict_examples)}] samples in dict_examples')
    # pprint(dict_examples)
    # logging.info('*'*50)
    # logging.info(f'Loaded {len(examples)} examples.')
    # logging.info(f'The number of samples with pronoun: {cnt_samples_with_prounoun}, percentage: {cnt_samples_with_prounoun/len(examples)*100:.2f}%')
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
