# RIGA at SemEval 2023 Task 2: MultiCoNER II (Multilingual Complex Named Entity Recognition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The repository describes RIGA team submission to MultiCoNER II.

## Getting started
1. Create a new environment
    ```bash
    python -m venv venv
    ```
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. Now your environment is ready. Next step is get the data from [MultiCoNER download page](https://multiconer.github.io/dataset). Put the data in `data` directory.
4. Convert the data using `parse_conll.py` script
    ```bash
   python parse_conll.py --source_path {specify a path to dataset in CoNNL format}
    ```
5. Start gathering context using `get_context.py` script. You'll need to specify your own API key and specifying the dataset split to use. You'll find a `TODO` comments in the file for a help
6. On step 5. each context is collected separately for easier navigation and not querying the same sentences multiple times in case of error.  
   On this step you'll need to merge all of them into a single file. Use `merge_context.py` script for this purpose. You'll also need to change the dataset split in order to merge contexts for all train/dev/test datasets.
7. The last step is NER model fine-tuning. You could run `python train.py --help` command to get all argument list.
   During the competition we used mainly either `distilbert-base-uncased` (66M parameters) or `xlm-roberta-large` models (558M parameters).
