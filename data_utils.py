import json
import csv
import numpy as np
import os
import sys
import math
from collections import defaultdict
import functools
import operator

from llm_utils import get_contextual_embeddings_batched, \
                      get_contextual_embeddings_batched_just_CLS_token, \
                      get_contextual_embeddings_batched_mean_hidden_tokens


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from random import sample, choices

from stackexchange import read_and_downsample_stackexchange, read_stackexchange_specific_doc_ids
from glove_utils import construct_average_glove_embeddings

def read_qnli(dataset_dir, concatenate_pairs):
    # 54 duplicates found by character histogram comparison after preprocessing
    # + 144 duplicates found by character histogram comparison after removing words not in GloVe/GloVe-SIF vocabularies
    with open('duplicate-doc-ids/qnli-duplicate-ids.json', 'r') as f:
        ids_to_skip = set(json.load(f))
    
    print('Skipping {} duplicates'.format(len(ids_to_skip)))

    # + skip the the 10% of training examples used for fine-tuning
    with open('./fine-tuning-doc-ids/roberta-large-qnli_fine-tuned.json', 'r') as f:
        ids_used_for_training = json.load(f)
    ids_to_skip.update(ids_used_for_training)

    print('Skipping {} total'.format(len(ids_to_skip)))

    ids = []
    text = []
    labels = []
    csv.field_size_limit(sys.maxsize)
    with open('{}/train.tsv'.format(dataset_dir), 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            if row[0] in ids_to_skip:
                continue
            ids.append(row[0])
            if concatenate_pairs:
                text.append(row[1] + ' ' + row[2])
            else:
                text.append((row[1], row[2]))
            if row[3] == 'entailment':
                labels.append(1)
            else:
                labels.append(0)
    return ids, text, labels


def read_mnli(dataset_dir, concatenate_pairs):
    # 268 duplicates found by character histogram comparison after preprocessing
    # + 721 duplicates found by character histogram comparison after removing words not in GloVe/GloVe-SIF vocabularies
    # with open('duplicate-doc-ids/mnli-duplicate-ids.json', 'r') as f:
    #     ids_to_skip = set(json.load(f))

    ids_to_skip = set()

    # print('Skipping {} duplicates'.format(len(ids_to_skip)))

    # # + skip the the 10% of training examples used for fine-tuning
    # with open('./fine-tuning-doc-ids/roberta-large-mnli_fine-tuned.json', 'r') as f:
    #     # get just the first id
    #     ids_used_for_training = [first for first, _, _ in json.load(f)]
    # ids_to_skip.update(ids_used_for_training)

    print('Skipping {} total'.format(len(ids_to_skip)))

    ids = []
    text = []
    labels = []
    csv.field_size_limit(sys.maxsize)
    with open('{}/train.tsv'.format(dataset_dir), 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            if row[0] in ids_to_skip:
                continue
            # there are three labels: entailment, contradiction, neutral
            # for now, leave out neutral to get a binary classfication task
            if row[10] == 'neutral':
                continue
            ids.append((row[0], row[1], row[2]))
            if concatenate_pairs:
                text.append(row[8] + ' ' + row[9])
            else:
                text.append((row[8], row[9]))
            if row[10] == 'entailment':
                labels.append(1)
            else:
                labels.append(0)
    return ids, text, labels

def read_snli(dataset_dir, concatenate_pairs):
    # 731 duplicates found by character histogram comparison after preprocessing
    # + 17 duplicates found by character histogram comparison after removing words not in GloVe/GloVe-SIF vocabularies
    with open('duplicate-doc-ids/snli-duplicate-ids.json', 'r') as f:
        ids_to_skip = set(json.load(f))

    print('Skipping {} duplicates'.format(len(ids_to_skip)))

    # + skip the the 10% of training examples used for fine-tuning
    with open('./fine-tuning-doc-ids/roberta-large-snli_fine-tuned.json', 'r') as f:
        ids_used_for_training = json.load(f)
    ids_to_skip.update(ids_used_for_training)

    print('Skipping {} total'.format(len(ids_to_skip)))
    
    ids = []
    text = []
    labels = []
    csv.field_size_limit(sys.maxsize)
    with open('{}/snli_1.0_train.txt'.format(dataset_dir), 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            if row[8] in ids_to_skip:
                continue
            # there are three labels: entailment, contradiction, neutral
            # for now, leave out neutral to get a binary classfication task
            if row[0] == 'neutral':
                continue
            row_id = row[8]
            if concatenate_pairs:
                row_text = row[5] + ' ' + row[6]
            else:
                row_text = (row[5], row[6])
            if row[0] == 'entailment':
                row_label = 1
            else:
                row_label = 0
            ids.append(row_id)
            text.append(row_text)
            labels.append(row_label)
    return ids, text, labels

def read_wnli(dataset_dir, concatenate_pairs):
    # 2 duplicates found by character histogram comparison after preprocessing
    # + 2 duplicates found by character histogram comparison after removing words not in GloVe/GloVe-SIF vocabularies
    with open('duplicate-doc-ids/wnli-duplicate-ids.json', 'r') as f:
        ids_to_skip = set(json.load(f))

    ids = []
    text = []
    labels = []
    csv.field_size_limit(sys.maxsize)
    with open('{}/train.tsv'.format(dataset_dir), 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            if row[0] in ids_to_skip:
                continue
            ids.append(row[0])
            if concatenate_pairs:
                text.append(row[1] + ' ' + row[2])
            else:
                text.append((row[1], row[2]))
            labels.append(int(row[3]))
    return ids, text, labels

def convert_doc_ids_to_indices_mnli(all_ids, specific_doc_ids):
    downsample_idxs = []
    all_ids = np.array(all_ids)
    for j, doc_id in enumerate(specific_doc_ids):
        # each doc_id is a tuple of 3 identifiers
        i = np.where(all_ids[:, 0] == doc_id[0])[0][0]
        downsample_idxs.append(i)
        found_doc_id = all_ids[i, :]
        assert (found_doc_id[0] == doc_id[0] and found_doc_id[1] == doc_id[1] and found_doc_id[2] == doc_id[2])
    return np.array(downsample_idxs)

def convert_doc_ids_to_indices_qnli_snli(all_ids, specific_doc_ids):
    downsample_idxs = []
    for j, doc_id in enumerate(specific_doc_ids):
        # they were saved as tuples, so convert to str
        doc_id_str = ''.join(doc_id)
        i = all_ids.index(doc_id_str)
        downsample_idxs.append(i)
        found_doc_id = all_ids[i]
        assert found_doc_id == doc_id_str
    return np.array(downsample_idxs)


# assumes a full list of vocabulary already exists in vocab_fn
def construct_bags_of_words(text, vocab_fn):
    print('Bagging words.')

    with open (vocab_fn, 'r') as f:
        vocabulary = json.load(f)

    print('Number of words in full vocabulary: {}'.format(len(vocabulary)))

    
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    X = vectorizer.fit_transform(text)
    features = vectorizer.get_feature_names()
    docs_by_features = X.toarray().astype(np.float64)
    print('Total number of word types: {}'.format(len(features)))

    return docs_by_features

def save_full_bag_of_words_vocab(text, vocab_fn):
    print('Bagging full dataset with full vocab.')
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text)
    features = vectorizer.get_feature_names()
    print('Total number of word types: {}'.format(len(features)))
    with open(vocab_fn, 'w') as f:
        json.dump(features, f)

def load_bag_of_words(dataset, dataset_dir, dataset_reader, downsample, sample_size, specific_doc_ids, doc_id_converter, vocab_fn):
    print('Loading {} dataset.'.format(dataset))

    ids, text, labels = dataset_reader(dataset_dir, concatenate_pairs=True)

    if not os.path.exists(vocab_fn):
        save_full_bag_of_words_vocab(text, vocab_fn)

    print('Converting labels to numpy.')
    labels = np.array(labels)

    print('Using {} documents.'.format(len(text)))

    # downsample to small balanced +/-
    if downsample:
        # if we haven't specified any documents, randomly sample a small balanced +/-
        if specific_doc_ids == '':
            print('Sampling {} examples.'.format(sample_size))
            positive_idxs = np.where(labels == 1)[0]
            negative_idxs = np.where(labels == 0)[0]
            n_to_keep_from_each_class = math.floor(sample_size/2)

            print('# positive examples: {}'.format(len(positive_idxs)))
            print('# negative examples: {}'.format(len(negative_idxs)))
            print('# to keep from each: {}'.format(n_to_keep_from_each_class))
            
            # random.sample needs a list, not a numpy array
            positive_choices = np.array(sample(list(positive_idxs), k=n_to_keep_from_each_class))
            negative_choices = np.array(sample(list(negative_idxs), k=n_to_keep_from_each_class))
            downsample_idxs = np.concatenate((positive_choices, negative_choices))
        
        else:
            # load the prescribed doc ids
            print('Using specific document ids.')
            with open(specific_doc_ids, 'r') as f:
                downsample_idxs = doc_id_converter(ids, json.load(f))

        # make sure every selected example is unique
        assert len(downsample_idxs) == len(set(downsample_idxs))


        #docs_by_features = docs_by_features[downsample_idxs, :]
        text = [text[i] for i in downsample_idxs]
        ids = [ids[i] for i in downsample_idxs]
        labels = labels[downsample_idxs]

    ids = np.array(ids)

    docs_by_features = construct_bags_of_words(text, vocab_fn)

    print('Using {} documents.'.format(ids.shape[0]))
    assert ids.shape[0] == docs_by_features.shape[0] and docs_by_features.shape[0] == labels.shape[0]

    # make sure there are no zero-length documents
    ids_to_keep = np.where(np.sum(docs_by_features, axis=1) >= 0)[0]
    assert ids_to_keep.shape[0] == docs_by_features.shape[0]

    return ids, docs_by_features, labels


def load_average_glove_embeddings(dataset, dataset_dir, dataset_reader, downsample, sample_size, specific_doc_ids, doc_id_converter, vocab_fn):
    print('Loading {} dataset.'.format(dataset))

    ids, text, labels = dataset_reader(dataset_dir, concatenate_pairs=True)
    
    print('Converting labels to numpy.')
    labels = np.array(labels)

    print('Using {} documents.'.format(len(text)))

    # downsample to small balanced +/-
    if downsample:
        # if we haven't specified any documents, randomly sample a small balanced +/-
        if specific_doc_ids == '':
            print('Sampling {} examples.'.format(sample_size))
            positive_idxs = np.where(labels == 1)[0]
            negative_idxs = np.where(labels == 0)[0]
            n_to_keep_from_each_class = math.floor(sample_size/2)

            print('# positive examples: {}'.format(len(positive_idxs)))
            print('# negative examples: {}'.format(len(negative_idxs)))
            print('# to keep from each: {}'.format(n_to_keep_from_each_class))
            
            # random.sample needs a list, not a numpy array
            positive_choices = np.array(sample(list(positive_idxs), k=n_to_keep_from_each_class))
            negative_choices = np.array(sample(list(negative_idxs), k=n_to_keep_from_each_class))
            downsample_idxs = np.concatenate((positive_choices, negative_choices))
        
        else:
            # load the prescribed doc ids
            print('Using specific document ids.')
            with open(specific_doc_ids, 'r') as f:
                downsample_idxs = doc_id_converter(ids, json.load(f))

        # make sure every selected example is unique
        assert len(downsample_idxs) == len(set(downsample_idxs))


        #docs_by_features = docs_by_features[downsample_idxs, :]
        text = [text[i] for i in downsample_idxs]
        ids = [ids[i] for i in downsample_idxs]
        labels = labels[downsample_idxs]

    ids = np.array(ids)

    docs_by_features = construct_average_glove_embeddings(text)

    print('Using {} documents.'.format(ids.shape[0]))
    assert ids.shape[0] == docs_by_features.shape[0] and docs_by_features.shape[0] == labels.shape[0]


    return ids, docs_by_features, labels


def load_contextual_embeddings(dataset, representation, dataset_dir, dataset_reader, downsample, sample_size, use_gpu, specific_doc_ids, doc_id_converter, llm_embedding_type):
    print('Loading {} dataset.'.format(dataset))

    ids, text, labels = dataset_reader(dataset_dir, concatenate_pairs=False)
    contexts = [c for c, _ in text]
    questions = [q for _, q in text]
    print('Using {} documents.'.format(len(ids)))

    # some datasets (e.g. WNLI) are small enough to not need downsampling
    if downsample:
        # if we haven't specified any documents, randomly sample a small balanced +/-
        if specific_doc_ids == '':
            labels = np.array(labels)
            print('Sampling {} examples.'.format(sample_size))
            positive_idxs = np.where(labels == 1)[0]
            negative_idxs = np.where(labels == 0)[0]
            n_to_keep_from_each_class = math.floor(sample_size/2)

            print('# positive examples: {}'.format(len(positive_idxs)))
            print('# negative examples: {}'.format(len(negative_idxs)))
            print('# to keep from each: {}'.format(n_to_keep_from_each_class))

            positive_choices = np.array(sample(list(positive_idxs), k=n_to_keep_from_each_class))
            negative_choices = np.array(sample(list(negative_idxs), k=n_to_keep_from_each_class))
            downsample_idxs = np.concatenate((positive_choices, negative_choices))
        else:
            # load the prescribed doc ids
            print('Using specific document ids.')
            with open(specific_doc_ids, 'r') as f:
                downsample_idxs = doc_id_converter(ids, json.load(f))
        
        # make sure we only have unique documents
        assert len(downsample_idxs) == len(set(downsample_idxs))

        print('Converting to numpy.')
        ids = np.array(ids)
        labels = np.array(labels)

        contexts = [contexts[i] for i in downsample_idxs]
        questions = [questions[i] for i in downsample_idxs]
        ids = ids[downsample_idxs]
        labels = labels[downsample_idxs]
    else:
        print('Converting to numpy.')
        ids = np.array(ids)
        labels = np.array(labels)

    # get the contextual embeddings
    if llm_embedding_type == 'concat_embedding':
        # get the full concatenated hidden embeddings across all tokens
        docs_by_features = get_contextual_embeddings_batched(contexts, questions, representation, use_gpu)
    elif llm_embedding_type == 'mean_embedding':
        # average all the tokens in the hidden layer
        docs_by_features = get_contextual_embeddings_batched_mean_hidden_tokens(contexts, questions, representation, use_gpu)
    else:
        # default: only get the hidden embedding for the CLS token to match how MLMs are finetuned
        docs_by_features = get_contextual_embeddings_batched_just_CLS_token(contexts, questions, representation, use_gpu)

    return ids, docs_by_features, labels


def load_bag_of_words_custom_data(ids, text, labels, dataset_dir):
    # concatenate text if each example contains multiple texts
    if isinstance(text[0], list):
        text = [' '.join(t) for t in text]

    vocab_fn = '{}/all-features.json'.format(dataset_dir)
    if not os.path.exists(vocab_fn):
        save_full_bag_of_words_vocab(text, vocab_fn)

    print('Using {} documents.'.format(len(text)))
    # make sure every selected example is unique
    #assert len(downsample_idxs) == len(set(downsample_idxs))

    docs_by_features = construct_bags_of_words(text, vocab_fn)



    # make sure there are no zero-length documents
    ids_to_keep = np.where(np.sum(docs_by_features, axis=1) >= 0)[0]
    assert ids_to_keep.shape[0] == docs_by_features.shape[0]

    return normalize_data(docs_by_features)

def load_contextual_embeddings_custom_data(text, dataset_dir, representation, use_gpu):
    assert isinstance(text[0], list)
    assert len(text[0]) == 2

    contexts = [c for c, _ in text]
    questions = [q for _, q in text]
    print('Using {} documents.'.format(len(text)))

    # get the contextual embeddings
    docs_by_features = get_contextual_embeddings_batched_just_CLS_token(contexts, questions, representation, use_gpu)

    return normalize_data(docs_by_features)

def load_custom_data(representation, ids, text, labels, dataset_dir, use_gpu):
    if representation == 'bag-of-words':
        return load_bag_of_words_custom_data(ids, text, labels, dataset_dir)
    elif representation == 'roberta-large':
        return load_contextual_embeddings_custom_data(text, dataset_dir, 'roberta-large', use_gpu)
    else:
        print('Representation not supported yet: {}'.format(representation))
        exit()


# requires binary labels
# data: [{id: id, data: [text1, text2, ...], label: label}]
# where label can be one of two strings
def read_raw_data(fn):
    ids = []
    text = []
    labels = []
    with open(fn, 'r') as f:
        for d in json.load(f):
            ids.append(d['id'])
            text.append(d['data'])
            labels.append(d['label'])
    # now convert labels to 0/1
    unique_labels = sorted(list(set(labels)))
    if len(unique_labels) != 2:
        print('Labels must be binary!')
        print('Labels in dataset: {}'.format(', '.join(unique_labels)))
        exit()
    label_to_index = {l: i for i, l in enumerate(unique_labels)}
    labels = np.array([label_to_index[l] for l in labels])
    return ids, text, labels

def normalize_data(docs_by_features):
    print('l2-normalizing documents.')
    normalizer = 1.0 / np.linalg.norm(docs_by_features, axis=1)
    docs_by_features *= normalizer[:, np.newaxis]
    return docs_by_features


def load_data(dataset, representation, dataset_dir, use_gpu, sample_size, specific_doc_ids, llm_embedding_type, args):
    vocab_fn = '{}/all-features.json'.format(dataset_dir)
    if dataset == 'qnli':
        dataset_reader = read_qnli
        downsample = True
        doc_id_converter = convert_doc_ids_to_indices_qnli_snli
    elif dataset == 'mnli':
        dataset_reader = read_mnli
        downsample = True
        doc_id_converter = convert_doc_ids_to_indices_mnli
    elif dataset == 'snli':
        dataset_reader = read_snli
        downsample = True
        doc_id_converter = convert_doc_ids_to_indices_qnli_snli
    elif dataset == 'wnli':
        dataset_reader = read_wnli
        # wnli is small enough to use the whole dataset
        downsample = False
        doc_id_converter = None
    elif dataset == 'stackexchange':
        stackexchanges = [args.stackexchange_1, args.stackexchange_2]
        label_type = args.stackexchange_label_type
        vocab_fn = '{}/all-features-{}-{}.json'.format(dataset_dir, args.stackexchange_1, args.stackexchange_2)
        if specific_doc_ids != '':
            dataset_reader = functools.partial(read_stackexchange_specific_doc_ids,
                                               specific_doc_ids_fn=specific_doc_ids,
                                               stackexchange_names=stackexchanges,
                                               label_type=label_type,
                                               n=sample_size,
                                               vocab_fn=vocab_fn)
        else:
            dataset_reader = functools.partial(read_and_downsample_stackexchange,
                                               stackexchange_names=stackexchanges,
                                               label_type=label_type,
                                               n=sample_size,
                                               vocab_fn=vocab_fn)

        # already downsampled
        downsample = False
        doc_id_converter = None
    else:
        print('Dataset not found: {}'.format(dataset))
        exit()

    if representation == 'bag-of-words':
        ids, docs_by_features, labels = load_bag_of_words(dataset, dataset_dir, dataset_reader, downsample, sample_size, specific_doc_ids, doc_id_converter, vocab_fn)
    elif representation == 'glove':
        ids, docs_by_features, labels = load_average_glove_embeddings(dataset, dataset_dir, dataset_reader, downsample, sample_size, specific_doc_ids, doc_id_converter, vocab_fn)
    elif representation == 'roberta' or representation == 'bert' or representation == 'roberta-large' \
      or representation == 'roberta-large-mnli' or representation == 'roberta-large-qnli' or representation == 'roberta-large-snli':
        ids, docs_by_features, labels = load_contextual_embeddings(dataset, representation, dataset_dir, dataset_reader, downsample, sample_size, use_gpu, specific_doc_ids, doc_id_converter, llm_embedding_type)
    else:
        print('Representation not found: {}'.format(representation))
        exit()

    docs_by_features = normalize_data(docs_by_features)
    return ids, docs_by_features, labels


