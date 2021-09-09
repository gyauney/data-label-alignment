import math
import numpy as np
import random
import os
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import matplotlib.patheffects as path_effects
import seaborn as sns
sns.set_style("whitegrid")

from random import choices

import csv
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import operator

from sklearn.utils.extmath import randomized_svd



def construct_gram_relu_kernel_vectorized(data):
    print('Constructing ReLU Gram matrix!')
    n = data.shape[0]

    print('\tSquaring data')
    dots = np.dot(data, np.transpose(data))
    # catch floating point errors that pop up when i == j
    print('\tMinning with 1')
    dots = np.minimum(dots, 1)
    print('\tDots:', dots.shape)

    print('\tTaking arccos')
    thetas = np.arccos(dots)
    print('\tThetas:', thetas.shape)

    # check for errors in thetas
    if np.isnan(thetas).any():
        print('\tError: NaN/Inf/-Inf found in thetas!')
        exit()

    # element-wise matrix operations
    H_infty = np.multiply(np.multiply(dots, (np.pi - thetas)), 1.0 / (2 * np.pi))

    return H_infty


# calculate or load H_infty
def get_H_infty(data, fn, recalculate=False, save=True):
    if recalculate or (not os.path.exists(fn)):
        H_infty = construct_gram_relu_kernel_vectorized(data)
        print('Saving H_infty with dimensions:', H_infty.shape)
        if save:
            np.save(fn, H_infty)
    else:
        print('Loading existing H_infty')
        H_infty = np.load(fn, mmap_mode='r')
    return H_infty

def get_inverse(H_infty, fn):
    if not os.path.exists(fn):
        print('Calculating inverse!')
        H_inverse = np.linalg.inv(H_infty)
        print('Saving inverse.')
        np.save(fn, H_inverse)
    else:
        print('Loading inverse.')
        H_inverse = np.load(fn, mmap_mode='r')
    return H_inverse

# random data says even if the files exist, the data was generated randomly, so recalculate everything
def get_eigenvalues_and_eigenvectors(H_infty, n, w_fn, v_fn, recalculate=False, save=True):    
    if recalculate or (not (os.path.exists(w_fn) and os.path.exists(v_fn))):
        print('Calculating eigenvalues/-vectors.')
        w, v = np.linalg.eig(H_infty)
        # TODO move sorting up here so everything's saved in the right order
        if save:
            np.save(w_fn, w)
            np.save(v_fn, v)
    else:
        print('Loading eigenvalues/-vectors.')
        w = np.load(w_fn)
        v = np.load(v_fn)
    # sort eigenvalues (and corresponding -vectors) from largest to smallest
    sorted_idxs = (np.argsort(w)[::-1])
    eigenvalues = w[sorted_idxs]
    eigenvectors = v[:, sorted_idxs]
    return eigenvalues, eigenvectors


def get_truncated_svd_inverse(H_infty, k):
    print('Getting approximate inverse from truncated SVD with k = {}'.format(k))
    U, S, V_transpose = randomized_svd(H_infty, n_components=k, n_iter=5, random_state=None)
    #U, S, V_transpose = np.linalg.svd(H_infty)
    return np.dot(np.dot(np.transpose(V_transpose), np.diag(np.reciprocal(S))), np.transpose(U))


def get_complexity_with_gram_inverse(H_inverse, labels, n):
    return np.sqrt(2.0 * np.dot(np.transpose(labels), np.dot(H_inverse, labels)) / float(n))

# produces identical answer as multiplication by inverse
# but doesn't require getting the inverse at all!!
def get_complexity_with_eigendecomposition(eigenvalues, eigenvectors, labels, n, lambda_inverse=None):
    # create diagonal matrix with inverse eigenvalues
    if lambda_inverse is None:
        lambda_inverse = np.identity(n) * (1.0 / eigenvalues)
    yT_Q = np.dot(np.transpose(labels), eigenvectors)
    QT_y = np.transpose(yT_Q)
    return np.sqrt(2.0 * np.dot(np.dot(yT_Q, lambda_inverse), QT_y) / float(n))

# plot histogram of DDC values! to learn about concentration possibility
# and highlight both the real labeling's complexity and the expectation
def plot_ddc_empirical_density(dir_name, title, random_complexities, reference_ddc):
    plt.figure(figsize=(6.4,2.4))
    ax = plt.axes()
    weights = np.ones_like(random_complexities) / len(random_complexities)
    # the color is an opaque version of the default blue with opacity 0.4 / 0.6
    ax.hist(random_complexities, bins=64, weights=weights, color='#aac8e1', edgecolor='#7eaed2')
    ax.scatter(reference_ddc, 0, s=200, marker='|', clip_on=False, color='black', zorder=10)
    ax.text(reference_ddc, -0.2, "DDC", transform=ax.get_xaxis_transform(), horizontalalignment='center', fontweight='bold')
    ax.scatter(np.mean(random_complexities), 0, s=200, marker='|', clip_on=False, color='black', zorder=10)
    ax.text(np.mean(random_complexities), -0.2, "E[DDC]", transform=ax.get_xaxis_transform(), horizontalalignment='center', fontweight='bold')
    savefig('{}/DDC-histogram_{}.pdf'.format(dir_name, title), bbox_inches='tight')
    plt.close()

# plot the empirical distribution function of DDC values
# and highlight both the real labeling's complexity and the expectation
def plot_ddc_empirical_distribution_function(dir_name, title, random_complexities, reference_ddc):
    sample_size = float(len(random_complexities))
    plt.figure(figsize=(6.4,2.4))
    ax = plt.axes()
    empirical_Fs, bins, patches = ax.hist(random_complexities, 64, density=True, histtype='step', cumulative=True, clip_on=False, zorder=10, color='#aac8e1', edgecolor='#7eaed2', linewidth=2)
    # delete right-most line from histogram to get a line plot
    patches[0].set_xy(patches[0].get_xy()[:-1])
    ax.scatter(reference_ddc, 0, s=100, marker='|', clip_on=False, color='black', zorder=10)
    ax.text(reference_ddc, -0.2, "DDC", transform=ax.get_xaxis_transform(), horizontalalignment='center', fontweight='bold')
    ax.scatter(np.mean(random_complexities), 0, s=100, marker='|', clip_on=False, color='black', zorder=10)
    ax.text(np.mean(random_complexities), -0.2, "E[DDC]", transform=ax.get_xaxis_transform(), horizontalalignment='center', fontweight='bold')
    # plot the actual estimated probability
    num_above = sum([int(c <= reference_ddc) for c in random_complexities])
    F_at_ddc = num_above/sample_size
    print('Number of random samples that DDC is above:', num_above)
    ax.scatter(reference_ddc, F_at_ddc, s=50, marker='x', clip_on=False, color='black', zorder=20)
    F_label = ax.text(reference_ddc, F_at_ddc + 0.05, '{:.4f}'.format(F_at_ddc), transform=ax.transData, horizontalalignment='center', fontweight='bold', zorder=30)
    F_label.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    ax.set_ylim([0, 1])

    # add an extra line out to DDC at 0 if DDC is way to the left
    # and update the data arrays for error bar construction
    if reference_ddc < bins[0]:
        ax.plot([reference_ddc, bins[0]], [0, 0], clip_on=False, zorder=10, color='#7eaed2', linewidth=2)
        bins = np.insert(bins, 0, reference_ddc)
        empirical_Fs = np.insert(empirical_Fs, 0, F_at_ddc)
    elif reference_ddc > bins[-1]:
        # add an extra line out to DDC at 1 if DDC is way to the right
        ax.plot([bins[-1], reference_ddc], [1, 1], clip_on=False, zorder=10, color='#7eaed2', linewidth=2)
        bins = np.append(bins, reference_ddc)
        empirical_Fs = np.append(empirical_Fs, F_at_ddc)

    # make the error bars flat! by doubling up points to get flat lines
    empirical_Fs = np.append(empirical_Fs, 1)
    flat_empirical_Fs = []
    for i in range(0, len(empirical_Fs) - 1):
        flat_empirical_Fs.extend([empirical_Fs[i], empirical_Fs[i]])
    empirical_Fs = np.array(flat_empirical_Fs)
    flat_bins = [bins[0]]
    for i in range(1, len(bins) - 1):
        flat_bins.extend([bins[i], bins[i]])
    flat_bins.append(bins[-1])
    bins = np.array(flat_bins)

    # calculate the error bars with a bound from the DKW inequality
    # via uniform convergence of the empirical distribution function
    delta = 0.001
    epsilon_dkw = (1 / (2.0 * sample_size)) * np.log(2/delta)
    lows = np.maximum(empirical_Fs - epsilon_dkw, 0)
    highs = np.minimum(empirical_Fs + epsilon_dkw, 1)
    # plot the error bars finally!
    ax.fill_between(bins, lows, highs, color='#aac8e1')

    # get confidence interval for estimated probability
    F_at_ddc_upper_bound = min(F_at_ddc + epsilon_dkw, 1)
    print('F(DDC)          = {:.8f}'.format(F_at_ddc))
    print('         ε_dkw <= {:.8f}'.format(epsilon_dkw))
    print('F(DDC) + ε_dkw <= {:.8f}'.format(F_at_ddc_upper_bound))

    savefig('{}/DDC-distribution-function_{}.pdf'.format(dir_name, title), bbox_inches='tight')
    plt.close()
    return F_at_ddc, F_at_ddc_upper_bound

def get_empirical_random_complexity(eigenvalues, eigenvectors, n, num_true, dir_name, title, reference_ddc, lambda_max, lambda_min):
    # simple Hoeffding proof to get within epsilon of the true expectation with high probability
    # i.e. with a 99% chance, the returned value will be at most 2 away from the true expectation

    # bound DDC in order to use hoffding properly
    ddc_max = np.sqrt(2.0 / (lambda_min))
    ddc_min = np.sqrt(2.0 / (lambda_max))
    print('Min eigenvalue: {:.4f}'.format(lambda_min))
    print('Max eigenvalue: {:.4f}'.format(lambda_max))
    print('Min DDC >= {:.4f}'.format(ddc_min))
    print('Max DDC <= {:.4f}'.format(ddc_max))
    print('Squared difference: {:.8f}'.format(math.pow(ddc_max - ddc_min, 2)))

    epsilon = 2
    delta = 0.01
    sample_size = math.ceil(((math.pow(ddc_max - ddc_min, 2) / (2.0 * math.pow(epsilon, 2))) * math.log(2.0 / delta)))
    print('Sample size:', sample_size)

    random_complexities = []
    for i in range(sample_size):
        if i % 10 == 0:
            print('{}/{} samples'.format(i, sample_size))
        
        # random balanced labels
        random_labels = np.array([-1] * n)
        positive_idxs = np.array(choices(range(n), k=num_true))
        random_labels[positive_idxs] = 1
    
        #random_complexity = np.sqrt(2 * np.dot(np.transpose(random_labels), np.dot(H_inverse, random_labels)) / float(n))
        random_complexity = get_complexity_with_eigendecomposition(eigenvalues, eigenvectors, random_labels, n, lambda_inverse = np.identity(n) * (1.0 / eigenvalues))
        random_complexities.append(random_complexity)

    # plot average DDC for increasing number of samples
    average_of_first_n = []
    for i in range(len(random_complexities)):
        prefix_avg = np.mean(random_complexities[:(i+1)])
        average_of_first_n.append(prefix_avg)
    plt.figure()
    ax = plt.axes()
    ax.plot(range(len(average_of_first_n)), average_of_first_n)
    ax.set_ylabel('Average random DDC')
    ax.set_xlabel('Number of random DDC samples')
    savefig('{}/random-DDC-prefix-averages_{}.pdf'.format(dir_name, title), bbox_inches='tight')
    plt.close()

    # plot histogram!
    # and highlight where the real complexity is
    plot_ddc_empirical_density(dir_name, title, random_complexities, reference_ddc)
    F_at_ddc, F_at_ddc_upper_bound = plot_ddc_empirical_distribution_function(dir_name, title, random_complexities, reference_ddc)


    # save results for future consolidated plotting
    with open('{}/{}_random-complexities.json'.format(dir_name, title), 'w') as f:
        json.dump(random_complexities, f)

    # TODO plot histogram of DDC values
    print('Empirical min DDC: {:.8f}'.format(np.min(random_complexities)))
    print('Empirical max DDC: {:.8f}'.format(np.max(random_complexities)))
    print('Average random DDC: {:.8f}'.format(np.mean(random_complexities)))
    print('Std dev random DDC: {:.8f}'.format(np.sqrt(np.var(random_complexities))))
    print('DDC of real labels: {:.8f}'.format(reference_ddc))
    return np.mean(random_complexities), np.sqrt(np.var(random_complexities)), epsilon, delta, F_at_ddc, F_at_ddc_upper_bound

# trace of inverse = sum of inverse eigenvalues
def get_expected_squared_ddc(eigenvalues, n):
    expected_squared_ddc = (2.0 / n) * np.sum([1.0 / l for l in eigenvalues])
    print('E[DDC^2] = {:.8f}'.format(expected_squared_ddc))
    print('E[DDC] <= sqrt(E[DDC^2]) = {:.8f}'.format(np.sqrt(expected_squared_ddc)))
    return np.sqrt(expected_squared_ddc)

if __name__ == '__main__':
    main()

