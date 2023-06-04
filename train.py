import re
import click
import mlflow
import numpy as np
from datasets import load_from_disk, DatasetDict, load_metric
from transformers import PfeifferConfig, AutoTokenizer, TrainingArguments, Trainer, AdapterTrainer, DataCollatorForTokenClassification, AutoModelForTokenClassification
from transformers.adapters import CompacterConfig


@click.command()
@click.option("--batch", default=32, help="Batch size")
@click.option("--model_name", default="xlm-roberta-large", help="Name of pretrained model")
@click.option("--lr", default=2e-5, help="Learning rate")
@click.option("--eval_steps", default=100, help="How often run evaluation")
@click.option("--num_epochs", default=50, help="Number of epochs")
@click.option("--esp", default=5, help="Early stopping patience")
@click.option("--name", default=None, help="Run name")
@click.option("--task_name", default=None, help="Adapter task name")
@click.option("--max_len", default=30, help="Adapter task name")
def run(batch, model_name, lr, eval_steps, num_epochs, esp, name, task_name, max_len):
    df = load_from_disk('data/en')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(examples):
        """
        After tokenization, a word is split into multiple tokens. This function assigns the same POS tag for every token of the word.
        """
        tokenized_inputs = tokenizer(examples["tokens"], examples["found_context"], truncation=True, max_length=250, padding='max_length', is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            is_second_sentence = False
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None or is_second_sentence:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx])

                if previous_word_idx and word_idx is None and not is_second_sentence:
                    is_second_sentence = True

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    label_list = df["train"].features[f"ner_tags"].feature.names

    tokenized_datasets = df.map(tokenize_and_align_labels, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    args = TrainingArguments(
        f"{model_name}-finetuned-full-data",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=num_epochs,
        weight_decay=0.1,
        save_steps=2000,
    )

    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p

        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l not in [-100, 73]]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l not in [-100, 73]]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def calculate_parameters(ann):
        """Calculate trainable parameters"""
        model_num_param = 0
        for p in ann.parameters():
            model_num_param += p.numel()

        return model_num_param

    def evaluate_dataset(df):
        predictions, labels, _ = trainer.predict(df)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l not in [-100, 73]]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l not in[-100, 73]]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        result_dict = {
            "test_precision": results["overall_precision"],
            "test_recall": results["overall_recall"],
            "test_f1": results["overall_f1"],
            "test_accuracy": results["overall_accuracy"],
        }

        tag_result = {f'tag_{k}_f1':v['f1'] for k, v in results.items() if type(v) is dict}

        result_dict.update(tag_result)
        return result_dict

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

    if task_name:
        adapter_config = PfeifferConfig()

        model.add_adapter(task_name, config=adapter_config)
        model.train_adapter(task_name)

        trainer = AdapterTrainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    run_adapter_name = f"_{task_name}" if task_name else ''
    with mlflow.start_run(run_name=f'{name}{run_adapter_name}_{model_name}'):
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('task_name', task_name)
        mlflow.log_param('model_size', calculate_parameters(model))

        trainer.train()
        trainer.evaluate()
        mlflow.log_metrics(evaluate_dataset(tokenized_datasets["test"]))

        model.save_all_adapters(f"./states/{task_name}")
        mlflow.log_artifacts(f'states/{task_name}', artifact_path="model")


if __name__ == '__main__':
    run()
