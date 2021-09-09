import math
import numpy as np
import random
import os
import json
from random import choices

from ddc_utils import get_eigenvalues_and_eigenvectors, \
                      get_complexity_with_eigendecomposition, \
                      get_H_infty, \
                      get_empirical_random_complexity, \
                      get_inverse, \
                      get_expected_squared_ddc
from data_utils import load_data

import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--representation', required=True, type=str)
    parser.add_argument('--dataset_dir', required=True, type=str)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--sample_size', required=True, type=int)
    parser.add_argument('--run_number', required=True, type=int)
    parser.add_argument('--specific_doc_ids', required=False, type=str, default='')
    # if using contextual embeddings, default embedding is just taking the [CLS] hidden embedding
    # to match BERT/RoBERTa classification/finetuning process in the original papers
    # other options: 1) concatenate all final hidden-layer token embeddings
    #                2) average all final hidden-layer token embeddings
    parser.add_argument('--concat_embedding', action='store_true', default=False)
    parser.add_argument('--mean_embedding', action='store_true', default=False)
    # the following are special arguments just for stackexchange datasets
    parser.add_argument('--stackexchange_1', required=False, type=str, default='')
    parser.add_argument('--stackexchange_2', required=False, type=str, default='')
    parser.add_argument('--stackexchange_label_type', required=False, type=str, default='')
    return parser.parse_args()

# data: a numpy array: n x num_features
# labels: a 1-d numpy array of length n
# dataset: string used for labeling output and temp files
def get_ddc(doc_ids, data, labels_0_or_1, dataset, representation, file_dir, output_dir, run_number, times):
    
    print('Run number: {}'.format(run_number))

    # save the elapsed time
    start = time.time()

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
    H_infty = get_H_infty(data, H_infty_fn, recalculate=recalc, save=save)

    # verify that the only gram matrix entries equal to 0.5 are on the diagonal
    # othewrwise there are duplicate examples
    parallel_vectors = np.argwhere(np.isclose(H_infty, 0.5, atol=1e-08) == True)
    num_parallel_vectors = parallel_vectors.shape[0]
    assert n == num_parallel_vectors
    end = time.time()
    times['H_infty_construction'] = end - start
    
    # get the eigendecomposition of the Gram matrix
    start = time.time()
    eigenvalues, eigenvectors =  get_eigenvalues_and_eigenvectors(H_infty, n, w_fn, v_fn, recalculate=recalc, save=save)
    end = time.time()
    times['eigendecomposition'] = end - start


    complexity = get_complexity_with_eigendecomposition(eigenvalues, eigenvectors, labels, n)
    print('DDC = {:.4f}'.format(complexity))
    expectation = get_expected_squared_ddc(eigenvalues, n)

    start = time.time()

    average_random, sigma_random, epsilon, delta, F_at_ddc, F_at_ddc_upper_bound = get_empirical_random_complexity(eigenvalues, eigenvectors, n, num_true, output_dir, '{}-{}_run-{}'.format(dataset, representation, run_number), complexity, eigenvalues[0], eigenvalues[-1])

    end = time.time()
    times['random_label_sampling'] = end - start
    times['total'] = np.sum([t for t in times.values()])
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

    
def main():
    args = parse_args()

    output_dir = './{}-{}_{}-sample-size'.format(args.dataset, args.representation.lower(), args.sample_size)    
    file_dir = './TEMP-FILES_{}-{}'.format(args.dataset, args.representation.lower())

    # stackexchange gets a special output_dir
    if args.dataset == 'stackexchange':
        if args.stackexchange_label_type not in ['name', 'year', 'ampm']:
            print('Unknown stackexchange label type: {}'.format(args.stackexchange_label_type))
            exit()
        output_dir = './{}-{}-{}-{}_{}-sample-size_{}-labels'.format(args.dataset, args.stackexchange_1, args.stackexchange_2, args.representation.lower(), args.sample_size, args.stackexchange_label_type)
        file_dir = './TEMP-FILES_{}-{}-{}-{}'.format(args.dataset, args.stackexchange_1, args.stackexchange_2, args.representation.lower())

    # can only choose one embedding type
    if args.concat_embedding and args.mean_embedding:
        print('You can only choose up to one of --concat_embedding and --mean_embedding.')
        exit()
    llm_embedding_type = None
    if args.concat_embedding:
        output_dir += '_concat'
        file_dir += '_concat'
        llm_embedding_type = 'concat_embedding'
    if args.mean_embedding:
        output_dir += '_mean'
        file_dir += '_mean'
        llm_embedding_type = 'mean_embedding'
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    start = time.time()
    times = {}

    doc_ids, docs_by_features, labels = load_data(args.dataset.lower(), args.representation.lower(), args.dataset_dir, args.gpu, args.sample_size, args.specific_doc_ids, llm_embedding_type, args)
    end = time.time()
    times['embeddings'] = end - start

    print('doc_ids:', doc_ids.shape)
    print('docs_by_features:', docs_by_features.shape)
    print('labels: {}/{} positive'.format(labels.shape, np.sum(labels)))

    # save the doc ids if newly sampled
    if args.specific_doc_ids == '' and args.dataset != 'mnist':
        with open('{}/{}-{}-sampled-doc-ids_run-{}.json'.format(output_dir, args.dataset, args.representation, args.run_number), 'w') as f:
            json.dump([tuple(doc_id) for doc_id in doc_ids], f)

    get_ddc(doc_ids, docs_by_features, labels, args.dataset, args.representation, file_dir, output_dir, args.run_number, times)





if __name__ == '__main__':
    main()






