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

def identify_pronouns(sentence):
    """
    Identify the first pronoun in the sentence.
    Returns a tuple of the pronoun and its index if found, otherwise ('', -1).
    """
    doc = nlp(sentence)
    for i, token in enumerate(doc):
        if token.text.lower() in PRONOUNS and token.pos_ == 'PRON':
            return (token.text, i)
    return ('', -1)


def read_augwow_examples(input_file):
    examples = []
    dict_examples = []
    logging.info(f'Reading AugWoW examples from {input_file} in read_augwow_examples().')
    
    with open(input_file, "r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader):
            item = json.loads(line) # Load JSON line - each sample in AugWoW is a JSON line
            
            ctx = ' '.join(item['context'])
            doc_tokens = ctx.split()  # doc_tokens is a list of tokens
            item_id = item['id']
            qas_id = item_id + '_' + str(idx) # Unique ID for each example for predicting the reference noun to each sample
            
            response = item['claim'].split('[RESPONSE]: ')[-1] # string after the '[RESPONSE]: ' tag in the claim
            found_pronoun, pronoun_index = identify_pronouns(response)
            if found_pronoun:
                question_text = construct_question_text(found_pronoun, response)
                is_impossible=False
            else:
                question_text = None
                is_impossible=True
            
            squad_example = SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=item['context'],
                
                doc_tokens=doc_tokens,
                found_pronoun=found_pronoun,
                pronoun_index=pronoun_index,
                
                is_impossible=is_impossible,

                answer_text='', # Not used in inference
                start_position_character=-1, # Not used in inference

                item = item, # Store the original item 
                orig_response=response
            )

            if idx % 10000 == 0: #augwow has 190k examples
                logging.info(f'[AugWoW index: {idx}] Found pronoun: {found_pronoun} at index: {pronoun_index} in response: {response}.')
                dict_example = squad_example.to_dict()
                logging.info(json.dumps(dict_example)+'\n')
                logging.info('*'*50)
            
            examples.append(squad_example)
            dict_examples.append(squad_example.to_dict())
    logging.info(f'Loaded {len(examples)} examples.')
    # with open(resolved_dir+'/augwow_found_pronouns.jsonl', 'w') as f:
    #     for item in examples:
    #         f.write(json.dumps(item) + '\n')

    return examples, dict_examples
    
def read_dialfact_examples(input_file):
    examples = []
    dict_examples = []
    logging.info(f'Reading Dialfact examples from {input_file} in read_dialfact_examples().')
    cnt_samples_with_prounoun = 0
    with open(input_file, "r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader):
            item = json.loads(line) # Load JSON line - each sample in AugWoW is a JSON line
            
            ctx = ' '.join(item['context']) # one string of the context
            doc_tokens = ctx.split()  # doc_tokens is a list of tokens
            item_id = item['id']
            qas_id = item_id + '_' + str(idx) # Unique ID for each example for predicting the reference noun to each sample
            
            response = item['response']
            found_pronoun, pronoun_index = identify_pronouns(response)
            print(f'found_pronoun: {found_pronoun}, pronoun_index: {pronoun_index}')
            pprint(item)
            print('*'*50)
            if found_pronoun:
                question_text = construct_question_text(found_pronoun, response)
                is_impossible=False
                cnt_samples_with_prounoun += 1
            else:
                question_text = None
                is_impossible=True
            
            squad_example = SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=item['context'],
                
                doc_tokens=doc_tokens,
                found_pronoun=found_pronoun,
                pronoun_index=pronoun_index,
                
                is_impossible=is_impossible,

                answer_text='', # Not used in inference
                start_position_character=-1, # Not used in inference

                item = item, # Store the original item 
                orig_response=response
            )

            if idx % 1000 == 0: #augwow has 190k examples
                logging.info('*'*50)
                logging.info(f'[Dialfact index: {idx}] Found pronoun: [{found_pronoun}] at index: [{pronoun_index}] in response: [{response}].')
                dict_example = squad_example.to_dict()
                logging.info(json.dumps(dict_example)+'\n')
                logging.info('*'*50)
            
            examples.append(squad_example)
            dict_examples.append(squad_example.to_dict())

    logging.info(f'Loaded {len(examples)} examples.')
    logging.info(f'The number of samples with pronoun: {cnt_samples_with_prounoun}, percentage: {cnt_samples_with_prounoun/len(examples)*100:.2f}%')

    return examples, dict_examples

def resolve_coref_with_augwow(input_file, coref_data, output_file):
    """
    Resolve coreference problems in AugWoW examples using the provided coreference data.
    Modified to include pronoun_index, found_pronoun, and predicted_pronoun in the output.
    """
    logging.info(f'Resolving coreference in AugWoW examples using coreference data from coref_data.')
    examples = read_augwow_examples(input_file)
    resolved_examples = []

    for item in examples:
        unique_id = item['id']
        # Ensure coref_data contains the unique_id
        if unique_id in coref_data:
            predicted_pronoun = coref_data[unique_id]['predicted_pronoun']  # Assuming coref_data structure
            item['predicted_pronoun'] = predicted_pronoun  # Add predicted pronoun to the item
            tokens = item['claim'].split()
            pronoun_index = item.get('pronoun_index', -1)
            if 0 <= pronoun_index < len(tokens):
                tokens[pronoun_index] = predicted_pronoun
                item['claim'] = ' '.join(tokens)
        # Even if not resolved, include the pronoun identification metadata
        else:
            item['predicted_pronoun'] = None  # Indicate no prediction was made
        resolved_examples.append(item)

    write_jsonl(resolved_examples, output_file)
    logging.info(f'Wrote resolved examples to {output_file}.')


def write_jsonl(data, file_path):
    """
    Write data to a JSONL file.
    """
    with open(file_path, 'w') as f:
        for i, item in enumerate(data):
            if i%10000 == 0: #augwow has 190k examples
                logging.info(f'Writing item {i} to {file_path}.')
            f.write(json.dumps(item) + '\n')
        logging.info(f'Finished writing {len(data)} items to {file_path}.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--coref_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)

    args = parser.parse_args()

    # Load coreference data
    try:
        with open(args.coref_file, "r", encoding="utf-8") as reader:
            coref_data = json.load(reader)
    except Exception as e:
        logging.error(f'Failed to load coreference data: {e}')
        return

    resolve_coref_with_augwow(args.input_file, coref_data, args.output_file)

if __name__ == "__main__":
    main()
