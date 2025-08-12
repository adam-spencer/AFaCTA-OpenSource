'''
finetune_data_generator.py

The purpose of this script is to generate finetuning input (`jsonl` files).
'''
import json
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

# AFaCTA system prompt
SYSTEM_PROMPT = 'You are an AI assistant who helps fact-checkers to identify fact-like information in statements.'

# AFaCTA Part 1 User Prompt
USER_PROMPT_TEMPLATE = '''Given the following <sentence> from a Tweet, does it contain any objective information?

<sentence>: '{sentence}'

Answer with Yes or No only.
'''


def process_file(data_location, f):
    df = pd.read_csv(data_location)
    raw_data = []
    print(df.columns)
    df.apply(
        lambda row:
        raw_data.append(
            (row['SENTENCES'], 'Yes' if row['Golden'] == 1 else 'No')),
        axis=1)

    for sentence, label in raw_data:
        user_content = USER_PROMPT_TEMPLATE.format(sentence=sentence)
        training_example = {
            'messages': [
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': user_content
                },
                {
                    'role': 'assistant',
                    'content': label
                }
            ]
        }
        f.write(json.dumps(training_example) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'input_files', help='CSV file(s) to generate finetuning data from.',
        nargs='+')
    parser.add_argument('--output-file', '-o', required=True)
    args = parser.parse_args()
    files = args.input_files
    output_file = args.output_file

    with open(output_file, 'w') as f:
        for file in files:
            process_file(file, f)

    print(f'Successfully created {output_file}')
