import math
import numpy as np
import random
import os
import json
from random import sample
import collections

from ddc_utils import get_eigenvalues_and_eigenvectors, \
                      get_complexity_with_eigendecomposition, \
                      get_H_infty, \
                      get_empirical_random_complexity, \
                      get_inverse, \
                      get_expected_squared_ddc
from data_utils import read_raw_data, load_custom_data
from plot_utils import plot_results

import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--dataset_fn', required=True, type=str)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--sample_size', required=True, type=int)
    parser.add_argument('--run_number', required=True, type=int)
    parser.add_argument('--specific_doc_ids', required=False, type=str, default='')
    return parser.parse_args()

# data: a numpy array: n x num_features
# labels: a 1-d numpy array of length n
# dataset: string used for labeling output and temp files
def get_ddc(doc_ids, data, labels_0_or_1, dataset, representation, file_dir, output_dir, run_number, times, H_infty=None):
    
    print('Run number: {}'.format(run_number))

    n = data.shape[0]
    num_features = data.shape[1]
    # switch to -1, 1 labels
    labels = labels_0_or_1
    labels[labels == 0] = -1
    num_true = len(labels[labels == 1])

    H_infty_fn = '{}/H_infty-{}_{}-features_{}-docs_run-{}.npy'.format(file_dir, dataset, num_features, n, run_number)
    H_inverse_fn = '{}/H_inverse-{}_{}-features_{}-docs_run-{}.npy'.format(file_dir, dataset, num_features, n, run_number)
    w_fn = '{}/eigenvalues-{}_{}-features_{}-docs_run-{}.npy'.format(file_dir, dataset, num_features, n, run_number)
    v_fn = '{}/eigenvectors-{}_{}-features_{}-docs_run-{}.npy'.format(file_dir, dataset, num_features, n, run_number)

    # do not save the Gram matrix or its eigendecomposition
    # and do not load any cached versions
    save = False
    recalc = True
    if H_infty is None:
        start = time.time()
        H_infty = get_H_infty(data, H_infty_fn, recalculate=recalc, save=save)
        end = time.time()
        times['H_infty_construction_{}'.format(representation)] = end - start
    else:
        print('Gram matrix already calculated from deduplication.')

    # verify that the only gram matrix entries equal to 0.5 are on the diagonal
    # othewrwise there are duplicate examples
    parallel_vectors = np.argwhere(np.isclose(H_infty, 0.5, atol=1e-08) == True)
    num_parallel_vectors = parallel_vectors.shape[0]
    assert n == num_parallel_vectors
    
    
    # get the eigendecomposition of the Gram matrix
    start = time.time()
    eigenvalues, eigenvectors =  get_eigenvalues_and_eigenvectors(H_infty, n, w_fn, v_fn, recalculate=recalc, save=save)
    end = time.time()
    times['eigendecomposition_{}'.format(representation)] = end - start


    complexity = get_complexity_with_eigendecomposition(eigenvalues, eigenvectors, labels, n)
    print('DDC = {:.4f}'.format(complexity))
    expectation = get_expected_squared_ddc(eigenvalues, n)

    start = time.time()

    average_random, sigma_random, epsilon, delta, F_at_ddc, F_at_ddc_upper_bound = get_empirical_random_complexity(eigenvalues, eigenvectors, n, num_true, output_dir, '{}-{}_run-{}'.format(dataset, representation, run_number), complexity, eigenvalues[0], eigenvalues[-1])

    end = time.time()
    times['random_label_sampling_{}'.format(representation)] = end - start
    print('Done with sampling random labels.')

    # save results
    fn = '{}/{}-{}-results_run-{}.json'.format(output_dir, dataset, representation, run_number)
    with open(fn, 'w') as f:
        results_json = {'dataset': dataset, 'representation': representation,
                   'sample_size': n, 'run_number': run_number,
                   'ddc': complexity,
                   'expectation_upper_bound': expectation,
                   'expectation_empirical': average_random,
                   'std_dev_empirical': sigma_random,
                   'epsilon': epsilon, 'delta': delta,
                   'empirical_F_at_ddc': F_at_ddc,
                   'F_at_ddc_upper_bound': F_at_ddc_upper_bound,
                   'empirical_distribution': 'balanced',
                   'elapsed_times': times}
        json.dump(results_json, f)
        json.dumps(results_json)
    print('Dataset: {} / Representation: {}'.format(dataset, representation))
    print('Saved results at: {}'.format(fn))
    return fn


# for working with custom data
def downsample(ids, text, labels, sample_size, ids_to_exclude=[], all_data_representations=[], verbose=True):
    positive_idxs = set(np.where(labels == 1)[0])
    negative_idxs = set(np.where(labels == 0)[0])
    n_to_keep_from_each_class = math.floor(sample_size/2)

    if len(ids_to_exclude):
        for idx in ids_to_exclude:
            if idx in positive_idxs:
                positive_idxs.remove(idx)
            if idx in negative_idxs:
                negative_idxs.remove(idx)

    if verbose:
        print('Excluding {} duplicate examples.'.format(len(ids_to_exclude)))
        print('# positive examples: {}'.format(len(positive_idxs)))
        print('# negative examples: {}'.format(len(negative_idxs)))
        print('# to keep from each: {}'.format(n_to_keep_from_each_class))
        
    # random.sample needs a list, not a numpy array
    positive_choices = np.array(sample(list(positive_idxs), k=n_to_keep_from_each_class))
    negative_choices = np.array(sample(list(negative_idxs), k=n_to_keep_from_each_class))
    downsample_idxs = np.concatenate((positive_choices, negative_choices))

    text = [text[i] for i in downsample_idxs]
    ids = [ids[i] for i in downsample_idxs]
    labels = labels[downsample_idxs]

    downsampled_datas = []
    for data in all_data_representations:
        downsampled_datas.append(data[downsample_idxs, :])

    return ids, text, labels, downsampled_datas

# for working with custom data
def downsample_truncate(ids, text, labels, ids_to_exclude=[], all_data_representations=[], all_Hs=[], verbose=True):
    positive_idxs = set(np.where(labels == 1)[0])
    negative_idxs = set(np.where(labels == 0)[0])

    if len(ids_to_exclude):
        for idx in ids_to_exclude:
            if idx in positive_idxs:
                positive_idxs.remove(idx)
            if idx in negative_idxs:
                negative_idxs.remove(idx)

    # truncate to smallest class AFTER excluding duplicates
    n_to_keep_from_each_class = min(len(positive_idxs), len(negative_idxs))

    if verbose:
        print('Excluding {} duplicate examples.'.format(len(ids_to_exclude)))
        print('# positive examples: {}'.format(len(positive_idxs)))
        print('# negative examples: {}'.format(len(negative_idxs)))
        print('# to keep from each: {}'.format(n_to_keep_from_each_class))
        
    # random.sample needs a list, not a numpy array
    positive_choices = np.array(sample(list(positive_idxs), k=n_to_keep_from_each_class))
    negative_choices = np.array(sample(list(negative_idxs), k=n_to_keep_from_each_class))
    downsample_idxs = np.concatenate((positive_choices, negative_choices))

    text = [text[i] for i in downsample_idxs]
    ids = [ids[i] for i in downsample_idxs]
    labels = labels[downsample_idxs]

    downsampled_datas = []
    for data in all_data_representations:
        downsampled_datas.append(data[downsample_idxs, :])

    downsampled_Hs = []
    for H in all_Hs:
        # first select rows, then select cols
        rows_to_keep = H[downsample_idxs, :]
        downsampled_Hs.append(rows_to_keep[:, downsample_idxs])

    return ids, text, labels, downsampled_datas, downsampled_Hs

def deduplicate(ids, data, times, representation):
    # do not save the Gram matrix
    # and do not load any cached versions
    save = False
    recalc = True
    start = time.time()
    H_infty = get_H_infty(data, '', recalculate=recalc, save=save)

    # verify that the only gram matrix entries equal to 0.5 are on the diagonal
    # othewrwise there are duplicate examples
    parallel_vectors = np.argwhere(np.isclose(H_infty, 0.5, atol=1e-8) == True)
    num_parallel_vectors = parallel_vectors.shape[0]
    print(num_parallel_vectors)
    duplicate_pairs = []
    if num_parallel_vectors > len(ids):
        for idxs in parallel_vectors:
            # ignore the diagonal:
            if idxs[0] == idxs[1]:
                continue
            # our canonical version is first index < second index
            # because the matrix is symmetric
            if idxs[0] > idxs[1]:
                continue
            duplicate_pairs.append(tuple(idxs))
    end = time.time()
    print('Time to deduplicate:', end - start)
    times['deduplicate_H_infty_construction_{}'.format(representation)] = end - start
    return duplicate_pairs, H_infty, times

# from https://stackoverflow.com/questions/48873107/detecting-equivalent-classes-with-python
class UnionFind:
    def __init__(self):
        self.leaders = collections.defaultdict(lambda: None)

    def find(self, x):
        l = self.leaders[x]
        if l is not None:
            l = self.find(l)
            self.leaders[x] = l
            return l
        return x

    def union(self, x, y):
        lx, ly = self.find(x), self.find(y)
        if lx != ly:
            self.leaders[lx] = ly

    def get_groups(self):
        groups = collections.defaultdict(set)
        for x in self.leaders:
            groups[self.find(x)].add(x)
        return list(groups.values())


def main():
    args = parse_args()

    representation_names = ['bag-of-words', 'roberta-large']

    # create output directories
    output_dirs = ['./{}-{}'.format(args.dataset, r.lower()) for r in representation_names] 
    file_dirs = ['./TEMP-FILES_{}-{}'.format(args.dataset, r.lower()) for r in representation_names]
    for output_dir, file_dir in zip(output_dirs, file_dirs):    
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

    times = {}

    # load raw text, ids, labels
    ids, text, labels = read_raw_data(args.dataset_fn)
    
    # sample evenly balanced documents
    ids, text, labels, _ = downsample(ids, text, labels, args.sample_size)

    # get different representations: bow and pre-trained embeddings
    all_data_representations = []
    all_Hs = []
    all_duplicate_id_pairs = []
    for representation, file_dir in zip(representation_names, file_dirs):
        print('Representation: {}'.format(representation))
        docs_by_features = load_custom_data(representation, ids, text, labels, file_dir, args.gpu)
        duplicate_id_pairs, H, times = deduplicate(ids, docs_by_features, times, representation)
        all_data_representations.append(docs_by_features)
        all_duplicate_id_pairs.extend(duplicate_id_pairs)
        all_Hs.append(H)

    # merge duplicate equivalence classes from all representations
    uf = UnionFind()
    for example_1, example_2 in all_duplicate_id_pairs:
        uf.union(example_1, example_2)
    equivalent_examples = uf.get_groups()

    # remove duplicates
    # remove all but one example (arbitrary: here it's random) from each set of equivalent examples
    ids_to_remove = [idx for duplicate_ids in equivalent_examples for idx in random.sample(list(duplicate_ids), len(duplicate_ids) - 1)]

    # downsample so each class is balanced:
    # truncate larger class to have same size as smaller class, so the classes are balanced
    # and remove the duplicate rows/cols in the gram matrices so we can reuse them
    ids, text, labels, all_data_representations, all_Hs = downsample_truncate(ids, text, labels, ids_to_remove, all_data_representations, all_Hs)
    print('Total number of duplicates removed: {}'.format(len(ids_to_remove)))

    # save doc ids
    if args.specific_doc_ids == '':
        for output_dir in output_dirs:
            with open('{}/{}-sampled-doc-ids_run-{}.json'.format(output_dir, args.dataset, args.run_number), 'w') as f:
                if isinstance(ids[0], list):
                    ids = [tuple(idx) for idx in ids]
                json.dump(ids, f)

    # run ddc on all representations
    # and reuse the gram matrices constructed for deduplication
    results_fns = []
    for representation, docs_by_features, H, output_dir, file_dir in zip(representation_names, all_data_representations, all_Hs, output_dirs, file_dirs):
        print('Getting DDC for representation: {}'.format(representation))
        results_fn = get_ddc(ids, docs_by_features, labels, args.dataset, representation, file_dir, output_dir, args.run_number, times, H)
        results_fns.append(results_fn)

    # make plots
    name = '{}_run-number-{}'.format(args.dataset, args.run_number)
    plot_results(results_fns, name)

    # print the report of settings
    print('-----------------------------------------')
    print('Dataset name: {}'.format(args.dataset))
    print('Dataset file: {}'.format(args.dataset_fn))
    print('Run number: {}'.format(args.run_number))
    print('Representations: {}'.format(', '.join(representation_names)))
    print('Original number of examples: {}'.format(args.sample_size))
    print('Number of examples after deduplicating: {}'.format(len(ids)))

    




if __name__ == '__main__':
    main()






