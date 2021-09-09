import xml.etree.ElementTree as ET
import argparse
import json
from collections import defaultdict
import random
import math
import os

from sklearn.feature_extraction.text import CountVectorizer

# found by checking for string-duplicate posts
ids_to_skip = {}
ids_to_skip['cstheory'] = set(['10663', '10669', '10983', '11192', '11956', '12014', '12077', '16275', '1741', '17646', '19799', '19891', '20575', '24906', '25479', '27826', '31049', '31378', '32119', '32443', '32695', '33721', '33932', '3546', '36016', '37662', '37739', '3785', '38087', '39461', '39662', '39802', '40486', '40608', '41608', '41695', '41722', '41773', '41828', '41963', '42155', '42165', '43825', '44109', '44290', '44366', '46028', '46449', '46870', '47476', '47527', '47609', '48066', '7956', '7957', '9926', '9928', '9930', '9932'])
ids_to_skip['bicycles'] = set(['44759', '47308', '44787', '45119', '52443', '69770', '73420', '44732', '44755', '61868', '72203', '42074', '46041', '44734', '44692', '45121', '47118', '46043', '10799'])
ids_to_skip['cs'] = set(['10264', '110950', '12011', '121454', '131959', '13601', '14336', '14386', '14387', '1765', '19357', '22945', '29681', '3058', '42143', '43556', '43557', '44059', '50966', '58232', '63234', '67156', '76144', '81327', '83181', '86923', '90084', '9112', '92', '986'])

##########
# from https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
##########

def save_full_vocab(dataset_dir, stackexchange_names, vocab_fn):
    print('Getting full vocabulary for {} and {} stackexchanges.'.format(*stackexchange_names))
    text = []
    for name in stackexchange_names:
        text.extend(read_single_stackexchange_text(dataset_dir, name))
    print('Bagging {} posts'.format(len(text)))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text)
    features = vectorizer.get_feature_names()
    print('Total number of word types: {}'.format(len(features)))
    with open(vocab_fn, 'w') as f:
         json.dump(features, f)

def read_single_stackexchange_text(dataset_dir, stackexchange_name):
    if stackexchange_name not in ids_to_skip:
        print('Find duplicates for {} stackexchange before running this!'.format(stackexchange_name))
        exit()

    root = ET.parse('{}/{}/Posts.xml'.format(dataset_dir, stackexchange_name)).getroot()
    all_posts = root.findall('row')

    texts = []
    for post in all_posts:
        text = strip_tags(post.get('Body')).replace('\n', ' ')
        idx = post.get('Id')

        # skip duplicates and short posts
        if idx in ids_to_skip[stackexchange_name]:
            continue
        if len(text) <= 20:
            continue

        texts.append(text)
    return texts

def read_and_downsample_single_stackexchange(dataset_dir, stackexchange_name, num_to_keep_from_each_split):
    if stackexchange_name not in ids_to_skip:
        print('Find duplicates for {} stackexchange before running this!'.format(stackexchange_name))
        exit()

    root = ET.parse('{}/{}/Posts.xml'.format(dataset_dir, stackexchange_name)).getroot()
    all_posts = root.findall('row')

    data_split = defaultdict(list)
    split_to_num = defaultdict(int)
    for post in all_posts:
        text = strip_tags(post.get('Body')).replace('\n', ' ')
        idx = post.get('Id')
        year = int(post.get('CreationDate').split('-')[0])
        hour = int(post.get('CreationDate').split('T')[1][:2])


        # skip duplicates and short posts
        if idx in ids_to_skip[stackexchange_name]:
            continue
        if len(text) <= 20:
            continue

        if year <= 2015 and hour < 12:
            split = '[2010, 2015] am'
        elif year <= 2015 and hour >= 12:
            split = '[2010, 2015] pm'
        elif year > 2015 and hour < 12:
            split = '[2016, 2021] am'
        elif year > 2015 and hour >= 12:
            split = '[2016, 2021] pm'
        else:
            print('Split not recognized: {} and {}'.format(year, hour))
            exit()

        data_split[split].append({'id': (stackexchange_name, idx),
                                  'year': year, 
                                  'hour': hour,
                                  'text': text})
        split_to_num[split] += 1

    print(split_to_num)
    # pick k from each split
    data = []
    for split in ['[2010, 2015] am', '[2010, 2015] pm',
                  '[2016, 2021] am', '[2016, 2021] pm']:
        print('{}: {}'.format(split, len(data_split[split])))
        data.extend(random.sample(data_split[split], k=num_to_keep_from_each_split))
    
    return data

def read_single_stackexchange_specific_doc_ids(dataset_dir, stackexchange_name, specific_doc_ids):
    if stackexchange_name not in ids_to_skip:
        print('Find duplicates for {} stackexchange before running this!'.format(stackexchange_name))
        exit()

    root = ET.parse('{}/{}/Posts.xml'.format(dataset_dir, stackexchange_name)).getroot()
    all_posts = root.findall('row')

    data = []
    for post in all_posts:
        text = strip_tags(post.get('Body')).replace('\n', ' ')
        idx = post.get('Id')
        year = int(post.get('CreationDate').split('-')[0])
        hour = int(post.get('CreationDate').split('T')[1][:2])

        # skip duplicates and short posts
        if idx in ids_to_skip[stackexchange_name]:
            continue
        if len(text) <= 20:
            continue

        named_id = (stackexchange_name, idx)
        if named_id not in specific_doc_ids:
            continue

        data.append({'id': named_id,
                     'year': year, 
                     'hour': hour,
                     'text': text})
    return data


def read_stackexchange_specific_doc_ids(dataset_dir, concatenate_pairs, specific_doc_ids_fn, stackexchange_names, label_type, n, vocab_fn):
    # load the ids
    with open(specific_doc_ids_fn, 'r') as f:
        specific_doc_ids = set([tuple(idx) for idx in json.load(f)])
    text = []
    ids = []
    labels = []
    for forum_number, name in enumerate(stackexchange_names):
        print('Sampling from {}'.format(name))
        sampled_posts = read_single_stackexchange_specific_doc_ids(dataset_dir, name, specific_doc_ids)
        for post in sampled_posts:
            ids.append(post['id'])
            text.append(post['text'])
            if label_type == 'name':
                labels.append(forum_number)
            elif label_type == 'year':
                before_2015 = int(post['year']) <= 2015
                labels.append(int(before_2015))
            elif label_type == 'ampm':
                ampm = (post['hour'] < 12)
                labels.append(int(ampm))
            else:
                print('Label type unknown: {}'.format(label_type))
                exit()
    print('# of positive examples: {}/{}'.format(sum(labels), len(labels)))
    assert len(ids) == len(text)
    assert len(text) == len(labels)
    assert sum(labels) == int(len(labels)/2)
    return ids, text, labels



def read_and_downsample_stackexchange(dataset_dir, concatenate_pairs, stackexchange_names, label_type, n, vocab_fn):
    
    if not os.path.exists(vocab_fn):
        save_full_vocab(dataset_dir, stackexchange_names, vocab_fn)

    text = []
    ids = []
    labels = []
    for forum_number, name in enumerate(stackexchange_names):
        print('Sampling from {}'.format(name))
        # sample n/4 posts from [2010, 2015] and n/4 from [2016, 2021]
        sampled_posts = read_and_downsample_single_stackexchange(dataset_dir, name, math.floor(n/8))
        for post in sampled_posts:
            ids.append(post['id'])
            text.append(post['text'])
            if label_type == 'name':
                labels.append(forum_number)
            elif label_type == 'year':
                before_2015 = int(post['year']) <= 2015
                labels.append(int(before_2015))
            elif label_type == 'ampm':
                ampm = (post['hour'] < 12)
                labels.append(int(ampm))
            else:
                print('Label type unknown: {}'.format(label_type))
                exit()
    assert len(ids) == len(text)
    assert len(text) == len(labels)
    assert sum(labels) == int(len(labels)/2)
    return ids, text, labels





