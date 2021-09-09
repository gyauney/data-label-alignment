# util for fine-tuning RoBERTa-large
# based on tutorial by Hugging Face

import numpy as np
import json
import os
import torch
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import argparse
from sklearn.metrics import accuracy_score



def read_all_qnli(dataset_dir, train_or_dev, fine_tune_ids=None):
    ids = []
    contexts = []
    questions = []
    labels = []

    if train_or_dev == 'train':
        fn = '{}/train.tsv'.format(dataset_dir)
    elif train_or_dev == 'dev':
        fn = '{}/dev.tsv'.format(dataset_dir)
    else:
        print('Unknown QNLI split:', train_or_dev)
        exit()

    # for some reason python's default csvreader can't parse dev.tsv
    with open(fn, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            row = line.strip('\n').split('\t')
            if train_or_dev == 'train' and row[0] not in fine_tune_ids:
                continue
            ids.append( row[0])
            contexts.append(row[1])
            questions.append(row[2])
            labels.append(row[3])

    return ids, contexts, questions, labels



def read_all_mnli(dataset_dir, train_or_dev, fine_tune_ids=None):
    ids = []
    contexts = []
    questions = []
    labels = []

    if train_or_dev == 'train':
        fn = '{}/train.tsv'.format(dataset_dir)
    elif train_or_dev == 'dev':
        fn = '{}/dev_matched.tsv'.format(dataset_dir)
    else:
        print('Unknown MNLI split:', train_or_dev)
        exit()

    with open(fn, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            row = line.strip('\n').split('\t')
            if train_or_dev == 'train' and row[0] not in fine_tune_ids:
                continue
            ids.append((row[0], row[1], row[2]))
            contexts.append(row[8])
            questions.append(row[9])
            labels.append(row[10])

    return ids, contexts, questions, labels



def read_all_snli(dataset_dir, train_or_dev, fine_tune_ids=None):
    ids = []
    contexts = []
    questions = []
    labels = []

    if train_or_dev == 'train':
        fn = '{}/snli_1.0_train.txt'.format(dataset_dir)
    elif train_or_dev == 'dev':
        fn = '{}/snli_1.0_dev.txt'.format(dataset_dir)
    else:
        print('Unknown SNLI split:', train_or_dev)
        exit()

    with open(fn, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            row = line.strip('\n').split('\t')
            # skip if there is no consensus for the label
            if row[0] == '-':
                continue
            if train_or_dev == 'train' and row[8] not in fine_tune_ids:
                continue
            ids.append(row[8])
            contexts.append(row[5])
            questions.append(row[6])
            labels.append(row[0])

    return ids, contexts, questions, labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--dataset_dir', required=True, type=str)
    parser.add_argument('--lr', required=False, type=float, default=2e-5)
    return parser.parse_args()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }

def finetune(cached_model_directory_name, lr, label2id,
             train_contexts, train_questions, train_labels,
             dev_contexts, dev_questions, dev_labels):
    model_name = 'roberta-large'  
    device_name = 'cuda'
    max_length = 128                                                        

    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    # # if labels aren't 0/1 already
    id2label = {idx: label for label, idx in label2id.items()}
    train_labels_encoded = [label2id[y] for y in train_labels]

    dev_labels_encoded = [label2id[y] for y in dev_labels]
    #train_labels_encoded = labels
    print(id2label)

    #train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    train_encodings = tokenizer(train_contexts, train_questions, padding='max_length', max_length=max_length, truncation=True)
    dev_encodings = tokenizer(dev_contexts, dev_questions, padding='max_length', max_length=max_length, truncation=True)
    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    dev_dataset = MyDataset(dev_encodings, dev_labels_encoded)

    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id)).to(device_name)

    training_args = TrainingArguments(
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        learning_rate=lr,              # initial learning rate for Adam optimizer
        output_dir=cached_model_directory_name,          # output directory
        logging_dir=cached_model_directory_name,            # directory for storing logs
        logging_steps=100,                # number of steps to output logging (set lower because of small dataset size)
        save_steps=10000
    )


    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset, 
        compute_metrics=compute_metrics      # our custom evaluation function 
    )

    trainer.train()

    trainer.save_model(cached_model_directory_name)

    eval_results = trainer.evaluate()
    print(eval_results)
    with open('{}/eval-results.json'.format(cached_model_directory_name), 'w') as f:
        json.dump(eval_results, f)

def main():

    args = parse_args()
    dataset = args.dataset.lower()

    cached_model_directory_name = './roberta-large-{}_fine-tuned'.format(dataset)
    if not os.path.exists(cached_model_directory_name):
        os.makedirs(cached_model_directory_name)

    if dataset == 'qnli':
        dataset_reader = read_all_qnli
        label2id = {'not_entailment':0,
                    'entailment':1
                   }
    elif dataset == 'mnli':
        dataset_reader = read_all_mnli
        label2id = {'contradiction': 0,
                    'entailment': 1,
                    'neutral': 2
                   }
    elif dataset == 'snli':
        dataset_reader = read_all_snli
        label2id = {'contradiction': 0,
                    'entailment': 1,
                    'neutral': 2
                   }
    else:
        print('Error: dataset not supported: {}'.format(dataset))
        exit()

    # train on a sampled 10% of the training set
    with open('fine-tuning-doc-ids/roberta-large-{}_fine-tuned.json'.format(dataset), 'r') as f:
        fine_tune_ids = json.load(f)

    train_ids, train_contexts, train_questions, train_labels = dataset_reader(args.dataset_dir, 'train', fine_tune_ids)
    dev_ids, dev_contexts, dev_questions, dev_labels = dataset_reader(args.dataset_dir, 'dev')

    print('Using {} documents.'.format(len(train_ids)))
    finetune(cached_model_directory_name, args.lr, label2id,
             train_contexts, train_questions, train_labels,
             dev_contexts, dev_questions, dev_labels)

if __name__ == '__main__':
    main()


