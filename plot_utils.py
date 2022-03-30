import math
import numpy as np
import random
import os
import json
# don't let matplotlib use xwindows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
sns.set_style("whitegrid")

sns.set_palette(sns.color_palette())

import pandas as pd
import glob
import argparse

def plot_results(result_fns, name):

    print('Plotting.')

    if not os.path.exists('./graphs'):
        os.makedirs('./graphs')

    # first consolidate results into dataframe
    datasets = []
    representations = []
    complexities = []
    expectation_empiricals = []
    std_dev_empiricals = []
    times_eigendecompositions = []
    times_embeddings = []
    times_random_labels = []
    for fn in result_fns:
        with open(fn, 'r') as f:
            d = json.load(f)
            datasets.append(d['dataset'])
            representations.append(d['representation'])
            complexities.append(float(d['ddc']))
            expectation_empiricals.append(float(d['expectation_empirical']))
            std_dev_empiricals.append(float(d['std_dev_empirical']))
    normalized_complexities = [c/m for c, m in zip(complexities, expectation_empiricals)]
    std_devs_from_expectation = [(c - m)/s for c, m, s in zip(complexities, expectation_empiricals, std_dev_empiricals)]

    results = pd.DataFrame({'Dataset': datasets,
                            'Representation': representations,
                            'DDC': complexities,
                            'E[DDC]': expectation_empiricals,
                            'DDC / E[DDC]': normalized_complexities,
                            '# of std devs DDC\nis above or below E[DDC]': std_devs_from_expectation
                           })

    # equivalent of figure 8a
    plt.figure()
    ax = sns.barplot(x='Representation', y='DDC', data=results)
    savefig('./graphs/ddc_{}.pdf'.format(name), bbox_inches='tight')
    plt.close()

    # equivalent of figure 8b
    plt.figure()
    ax = sns.barplot(x='Representation', y='E[DDC]', data=results)
    savefig('./graphs/ddc-random_{}.pdf'.format(name), bbox_inches='tight')
    plt.close()

    # equivalent of figure 8c
    plt.figure()
    ax = sns.barplot(x='Representation', y='DDC / E[DDC]', data=results)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    savefig('./graphs/ddc-ratio_{}.pdf'.format(name), bbox_inches='tight')
    plt.close()

    # equivalent of figure 8d
    plt.figure()
    ax = sns.barplot(x='Representation', y='# of std devs DDC\nis above or below E[DDC]', data=results)
    ax.axhline(linewidth=1, color='#cccccc')
    savefig('./graphs/ddc-z-score_{}.pdf'.format(name), bbox_inches='tight')
    plt.close()

