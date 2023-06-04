import os
import openai
from tqdm import tqdm
import json
import time
import logging
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Execute command
# python get_context.py > log1.txt 2>&1 &

split = 'en_test'  # TODO: specify what dataset split to use for context gathering
start = 0
limit = 20000
filepath = f'data/{split}.txt'
with open(filepath, 'r') as f:
    lines = f.readlines()

lines = [l.strip() for l in lines][start:limit]


openai.api_key = '...'  # TODO: update me


logger.info(f'Used token: {openai.api_key}')
logger.info(f'Total sentences to process: {len(lines)}')
logger.info(f'Start: {start}, Limit: {limit}, Split: {split}')

def request_context(query, i=0):
    try:
        response = openai.Completion.create(model="text-davinci-003",
                                            prompt=query,
											temperature=0.2,
											max_tokens=96,
											top_p=1,
											frequency_penalty=0.5,
											presence_penalty=0)['choices'][0]['text'].strip()
    except (openai.error.RateLimitError,
	    	openai.error.APIError,
	    	openai.error.Timeout,
	    	openai.error.APIConnectionError,
	    	openai.error.ServiceUnavailableError) as error:
        if i < 5:
            i += 1
            time.sleep(15)
            return request_context(query, i)
        raise error

    return response


for i, text in enumerate(tqdm(lines)):
    file_name = start + i
    out_file = f"data/{split}/{file_name}.json"
    if os.path.isfile(out_file):
        continue

    context_query = f"Give more context about named entities in the text:\n{text}"

    found_context = request_context(context_query)

    result = {
        "text": text,
        "context_query": context_query,
        "found_context": found_context
    }
    json.dump(result, open(out_file, 'w'))
