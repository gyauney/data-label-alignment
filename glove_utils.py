import numpy as np
import json
import os
from sklearn.feature_extraction.text import CountVectorizer

def load_glove_vocabulary():
    # step 0: check for cached numpy vectors
    cached_fn = 'GloVe/glove.840B.300d.npy'
    cached_word_to_idx_fn = 'GloVe/word_to_idx.json'
    if os.path.exists(cached_fn) and os.path.exists(cached_word_to_idx_fn):
        with open(cached_word_to_idx_fn, 'r') as f:
            word_to_idx = json.load(f)
        return set(word_to_idx.keys())

    words = set()
    with open('GloVe/glove.840B.300d.txt', 'r') as f:
        for line in f.read().splitlines():
            if len(line) == 0:
                continue
            split_line = line.split(' ')
            word = split_line[0]
            nums = split_line[1:]
            assert len(nums) == 300
            words.add(word)
    
    return words

# return glove embeddings as a (num words) x (num embedding dimensions) numpy array
def load_glove_vectors():
    # step 0: check for cached numpy vectors
    cached_fn = 'GloVe/glove.840B.300d.npy'
    cached_word_to_idx_fn = 'GloVe/word_to_idx.json'
    if os.path.exists(cached_fn) and os.path.exists(cached_word_to_idx_fn):
        glove = np.load(cached_fn, mmap_mode='r')
        with open(cached_word_to_idx_fn, 'r') as f:
            word_to_idx = json.load(f)
        return glove, word_to_idx

    # step 1: load the existing vectors
    word_to_idx = {}
    data = []
    with open('GloVe/glove.840B.300d.txt', 'r') as f:
        idx = 0
        for line in f.read().splitlines():
            if len(line) == 0:
                continue
            split_line = line.split(' ')
            word = split_line[0]
            nums = split_line[1:]
            assert len(nums) == 300
            word_to_idx[word] = idx
            data.append([float(d) for d in nums])
            idx += 1
    glove = np.array(data)
    print(glove.shape)
    print('saving')
    np.save(cached_fn, glove)
    print('done saving')
    with open(cached_word_to_idx_fn, 'w') as f:
        json.dump(word_to_idx, f)

    return glove, word_to_idx
    # step 2: add in random vectors for vocab words that aren't already present



def construct_average_glove_embeddings(text):
    print('Getting average glove embeddings.')

    # step 0: load all glove word embeddings
    glove, word_to_idx = load_glove_vectors()
    num_hidden_dims = glove.shape[1]

    # step 1: preprocess and tokenize just like bag-of-words
    vectorizer = CountVectorizer()
    preprocessor = vectorizer.build_preprocessor()
    tokenizer = vectorizer.build_tokenizer()

    # step 2: for each sentence, average over word embeddings
    docs_by_features = np.zeros((len(text), num_hidden_dims))
    for s, sentence in enumerate(text):
        if s % 100 == 0:
            print(s, sentence)
        words = tokenizer(preprocessor(sentence))
        words_by_hiddendims = np.zeros((len(words), num_hidden_dims))
        num_skipped = 0
        for w, word in enumerate(words):
            if word not in word_to_idx:
                num_skipped += 1
                continue
            words_by_hiddendims[w, :] = glove[word_to_idx[word], :]
        assert num_skipped < len(words)
        docs_by_features[s, :] = np.mean(words_by_hiddendims, axis=0)

    return docs_by_features



