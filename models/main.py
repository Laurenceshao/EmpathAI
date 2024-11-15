import huggingface_hub as hfhub
import os
from sklearn.utils import class_weight as sk_class_weight
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments)
import datasets
import click

from util import get_device


DATASET_PATH = "data/500_Reddit_users_posts_labels.csv"
DATASET_LABELS = ['Supportive', 'Ideation', 'Behavior', 'Attempt', 'Indicator']


def dataset_metadata():
    df = pd.read_csv(DATASET_PATH)
    class_weights = sk_class_weight.compute_class_weight(
        'balanced', classes=np.unique(df['label']), y=df['label'])
    return {
        'class_weights': torch.tensor(class_weights, dtype=torch.float32)
    }


def dataset_factory(
    val_ratio: int = 0.1
):
    metadata = dataset_metadata()
    dataset = datasets.load_dataset("csv", 
        data_files=DATASET_PATH)['train'].train_test_split(test_size=val_ratio)
    return dataset['train'], dataset['test'], metadata


def model_factory():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    classifer = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(DATASET_LABELS))
    return tokenizer, classifer


@click.command()
@click.option('--batch_size', default=16, type=int)
@click.option('--num_epochs', default=100, type=int)
@click.option('--learning_rate', default=1e-3, type=float)
@click.option('--weight_decay', default=1e-2, type=float)
@click.option('--output_dir', default='output', type=str)
@click.option('--logging_steps', default=10, type=int)
def train(
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    output_dir: str,
    logging_steps: int,
):
    device = get_device(use_gpu=True)
    print(f"Using device: {device}")

    tokenizer, classifier = model_factory()
    classifier = classifier.to(device)

    train_dataset, val_dataset, metadata = dataset_factory()
    
    # tokenization
    train_dataset = train_dataset.map(
        lambda x: tokenizer(x['text'], padding="max_length", truncation=True), 
        batched=True)
    val_dataset = val_dataset.map(
        lambda x: tokenizer(x['text'], padding="max_length", truncation=True), 
        batched=True)
    

    # setup training
    def compute_loss(output, labels, 
        return_outputs=False, num_items_in_batch=None,
        weight=metadata['class_weights']
    ):
        pred = F.log_softmax(output.logits, dim=-1)
        pred = pred.cpu() # cpu only
        labels = labels.cpu() # cpu only
        loss = F.cross_entropy(pred, labels.long(), weight=weight)
        loss = loss.to('mps') # cpu only
        return loss

    trainer = Trainer(
        model=classifier,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=logging_steps,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_loss_func=compute_loss,
    )
    trainer.train()

    # save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))

    # evaluation
    eval_result = trainer.evaluate()
    with open(os.path.join(output_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f)




if __name__ == '__main__':
    train()
