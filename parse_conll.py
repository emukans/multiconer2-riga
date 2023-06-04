import os
import click
import json

import itertools
import logging
from datasets import load_dataset, ClassLabel
from nltk.tokenize.treebank import TreebankWordDetokenizer

logger = logging.getLogger()

tags = [a.strip() for a in """Facility, OtherLOC, HumanSettlement, Station, VisualWork, MusicalWork, WrittenWork, ArtWork, Software, OtherCW, MusicalGRP, PublicCORP, PrivateCORP, OtherCORP, AerospaceManufacturer, SportsGRP, CarManufacturer, TechCORP, ORG,
Scientist, Artist, Athlete, Politician, Cleric, SportsManager, OtherPER, Clothing, Vehicle, Food, Drink, OtherPROD,
Medication/Vaccine, MedicalProcedure, AnatomicalStructure, Symptom, Disease""".split(',')]

tags_i = ["I-" + t for t in tags]
tags_b = ["B-" + t for t in tags]
tags = tags_b + tags_i

tag_mapping = dict()
for i, t in enumerate(tags):
    tag_mapping[t] = i

tag_mapping['O'] = max(tag_mapping.values()) + 1

tag_mapping_lower = dict()
for i, t in enumerate(tags):
    tag_mapping_lower[t.lower()] = i

tag_mapping_lower['O'.lower()] = max(tag_mapping_lower.values()) + 1


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-":  # or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False


def get_ner_reader(data_path):
    data_file = open(data_path, "r")
    print(data_path)
    for is_divider, lines in itertools.groupby(data_file, _is_divider):
        if is_divider:
            continue
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '').replace('\u200b', '') for line in lines]

        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None
        fields = [line.split() for line in lines if not line.startswith('# id')]
        fields = [list(field) for field in zip(*fields)]

        yield fields, metadata


class CoNLLReader:
    def __init__(self):
        self.instances = []

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data_path, has_tokens):
        logger.info('Reading file {}'.format(data_path))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data_path):
            result = {'tokens': fields[0]}

            if has_tokens:
                ner_tags = [list(tag_mapping.keys())[tag_mapping_lower[t.lower()]] for t in fields[-1]]
                result['ner_tags'] = ner_tags

            self.instances.append(result)

        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), data_path))


@click.command()
@click.option("--source_path", default=None, help="Data source path")
@click.option("--has_tokens", default=True, help="Has tokens")
def run(source_path, has_tokens):
    reader = CoNLLReader()
    reader.read_data(source_path, has_tokens)

    json_path = source_path.replace('conll', 'json')
    json.dump(reader.instances, open(json_path, 'w'))

    dataset = load_dataset('json', data_files=json_path, split="train")

    if has_tokens:
        new_features = dataset.features.copy()
        new_features['ner_tags'].feature = ClassLabel(names=list(tag_mapping.keys()))
        dataset = dataset.cast(new_features)

    dataset_path = os.path.dirname(source_path)
    dataset.save_to_disk(dataset_path)

    detokenizer = TreebankWordDetokenizer()

    text_list = [detokenizer.detokenize(text) + "\n" for text in dataset['tokens']]
    sentences_texts = source_path.replace('.conll', '_sentences.txt')
    with open(sentences_texts, 'w') as f:
        f.writelines(text_list)


if __name__ == '__main__':
    run()
