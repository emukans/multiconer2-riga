import json
import os

dir_path = 'data/test'  # TODO: specify the dataset split
full_data = []
for file_name in os.listdir(dir_path):
    entry = json.load(open(os.path.join(dir_path, file_name), 'r'))
    full_data.append(entry)

json.dump(full_data, open('test_context.json', 'w'))
