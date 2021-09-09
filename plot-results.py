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

if not os.path.exists('./graphs'):
    os.makedirs('./graphs')

parser = argparse.ArgumentParser()
parser.add_argument('--sample_size', required=True, type=int)
args = parser.parse_args()

# collect all the results JSON files for this sample size
fns = [fn for fn in glob.glob('*NLI-*_{}-sample-size/*results*.json'.format(args.sample_size))]

print('Consolidating results into dataframe.')
datasets = []
representations = []
complexities = []
expectation_empiricals = []
std_dev_empiricals = []
times_eigendecompositions = []
times_embeddings = []
times_random_labels = []
for fn in fns:
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

print('Plotting.')
colors = {'bag-of-words': '#1d78b4',
          'glove': '#7eaed2',
          'BERT': '#e1802b',
          'RoBERTa-large': '#d62629',
          'RoBERTa-large-mnli': '#9367bd',
          'RoBERTa-large-qnli': '#d584bd',
          'RoBERTa-large-snli': '#bfa8c9'
          }

hue_order = ['bag-of-words',
             'glove',
             'BERT',
             'RoBERTa-large',
             'RoBERTa-large-mnli',
             'RoBERTa-large-qnli',
             'RoBERTa-large-snli']

# Figure 8a
plt.figure(figsize=(10,2.4))
ax = sns.barplot(x='Dataset', y='DDC', hue='Representation', data=results, palette=colors, hue_order=hue_order)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.get_legend().remove()
savefig('./graphs/figure-8a_ddc.pdf', bbox_inches='tight')
plt.close()

# Figure 8b
plt.figure(figsize=(7,2.4))
ax = sns.barplot(x='Dataset', y='E[DDC]', hue='Representation', data=results, palette=colors, hue_order=hue_order)
h1 = matplotlib.lines.Line2D([], [], color="none")
l1 = matplotlib.patches.Patch(facecolor=colors['bag-of-words'])
l2 = matplotlib.patches.Patch(facecolor=colors['glove'])
h2 = matplotlib.lines.Line2D([], [], color="none")
l4 = matplotlib.patches.Patch(facecolor=colors['BERT'])
l5 = matplotlib.patches.Patch(facecolor=colors['RoBERTa-large'])
h3 = matplotlib.lines.Line2D([], [], color="none")
l7 = matplotlib.patches.Patch(facecolor=colors['RoBERTa-large-mnli'])
l8 = matplotlib.patches.Patch(facecolor=colors['RoBERTa-large-qnli'])
l9 = matplotlib.patches.Patch(facecolor=colors['RoBERTa-large-snli'])
ax.legend((h1,l1,l2, h2, l4,l5, h3, l7,l8,l9),(r'$\bf{Baselines}$', 'bag-of-words', 'glove', r'$\bf{Pre–trained}$', 'BERT', 'RoBERTa-large', r'$\bf{Fine–tuned}$', 'RoBERTa-large-mnli', 'RoBERTa-large-qnli', 'RoBERTa-large-snli'),
           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
savefig('./graphs/figure-8b_ddc-random.pdf', bbox_inches='tight')
plt.close()

# Figure 8c
plt.figure(figsize=(10,2.4))
ax = sns.barplot(x='Dataset', y='DDC / E[DDC]', hue='Representation', data=results, palette=colors, hue_order=hue_order)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.get_legend().remove()
savefig('./graphs/figure-8c_ddc-ratio.pdf', bbox_inches='tight')
plt.close()

# Figure 8d
plt.figure(figsize=(10,2.4))
ax = sns.barplot(x='Dataset', y='# of std devs DDC\nis above or below E[DDC]', hue='Representation', data=results, palette=colors, hue_order=hue_order)
ax.axhline(linewidth=1, color='#cccccc')
ax.get_legend().remove()
savefig('./graphs/figure-8d_ddc-z-score.pdf', bbox_inches='tight')
plt.close()

