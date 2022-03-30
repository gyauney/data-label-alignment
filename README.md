# data-label-alignment

Code for "[Comparing Text Representations: A Theory-Driven Approach](https://aclanthology.org/2021.emnlp-main.449.pdf)" (EMNLP 2021).

This code requires Python 3 (version >= 3.6). Use pip to install requirements:

```
pip install -r requirements.txt
```

## How to get data-label alignment of a new dataset

### 1. Input
Format your text dataset into a single JSON list that contains text examples and labels.
Each element of the list corresponds to an example in the dataset and must be a dictionary with three fields: `"id"`, `"data"`, and `"label"`.
Here's an example of the types for a single-example json file:
```[{"id": <string>, "data": [<string>, <string>, ...], "label": <string>}]```

- `"id"` can be either a string or a list of strings.
- `"data"` can also be either a string or a list of two strings.
If the example contains multiple pieces of text (e.g. as in NLI tasks), then `"data"` can point to a list of strings.
- `"label"` should be a string, and there must be two strings total across all examples.
For example, a binarized version of the MNLI task would have `"entailment"` and `"contradiction"` as the two label strings.

An example of 1000 documents from the MNLI dataset can be found in the file `mnli-formatted-sample.json`.

### 2. Running data-label alignment

You'll use `run-on-your-own-data.py` in this step. First, an explanation of arguments:
- `--sample_size <integer>`: determines how many examples are subsampled from the dataset for analysis. 
- `--dataset_fn <string>`: the name of the JSON file you constructed in step 1.
- `--dataset <string>`: the name of the dataset (for saving output)
- `--run_number <integer>`: an integer to identify where to store the results of this run. You can re-run this script with different run numbers to get different subsamples of a large dataset.
- `--gpu`: when present, the language model representations are gotten using the GPU. Remove if you don't have a GPU on your machine.

Here's an example of running the script on the included small version of the formatted MNLI dataset, with 1000 examples sampled:
```
python run-on-your-own-data.py --sample_size 1000 \
                               --dataset_fn mnli-formatted-sample.json \
                               --dataset MNLI \
                               --run_number 1 \
                               --gpu
```
You can run this command to see some example output graphs.

By default, the script compares the data-dependent complexities of two representations: 1) bag-of-words and 2) RoBERTa-large pre-trained embeddings. It first subsamples a chosen number of examples (with an equal number from each class) and then removes any examples that are duplicates under any of the considered representations. It is important that the classes remain balanced, so we also truncate the larger class after deduplication so that both classes have the same number of examples.

### 3. Output

Four summary graphs are saved in the `graphs` subdirectory:

- `ddc.pdf`: the data-dependent complexity of the true labels for each representation
- `random.pdf`: the data-dependent complexity of the sampled random labels for each representation
- `ddc-ratio.pdf`: the ratio between the true and random data-dependent complexities for each representation
- `ddc-z-score.pdf`: the distance of the true labeling's data-dependent complexity from the expected complexity under random labelings, measured in number of standard deviations of the random complexities

There are also more detailed graphs for each representation that show the distributions of data-dependent complexities from uniform random labels. These are saved in directories with named for the dataset and representation (they have the form `<dataset>-<representation name>`). For interpretations of all of the kinds of graphs, see the discussion of Figure 7 in Section 4 of the [paper](https://aclanthology.org/2021.emnlp-main.449.pdf).



## Reproducing results from the paper

These are instructions for reproducing the paper's main results on natural language inference datasets.


### 1. Download NLI datasets:

- MNLI (298.3 MB): https://dl.fbaipublicfiles.com/glue/data/MNLI.zip 
- QNLI (10.1 MB): https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip
- SNLI (90.2 MB): https://nlp.stanford.edu/projects/snli/snli_1.0.zip 
- WNLI (28.3 KB): https://dl.fbaipublicfiles.com/glue/data/WNLI.zip

We recommend saving these inside the current directory.

### 2. *Optional:* fine-tune RoBERTa-large on subsets of MNLI, QNLI, SNLI.

Fine-tuning will allow you to use our method to evaluate fine-tuned MLM contextual embeddings. For example:

```
python finetune.py \
    --dataset QNLI \
    --dataset_dir ./QNLI \
    --lr 2e-5
```

The options for the `--dataset` flag are `MNLI`, `QNLI`, and `SNLI`. The path passed to the `--dataset_dir` flag must match the path you downloaded data to in step 2. The `--lr` flag specifies the initial learning rate.

You can skip this step if you wish to only evaluate baseline non-contextual and pre-trained MLM contextual embeddings.

### 3. *Optional:* download GloVe embeddings.

Download pre-trained GloVe embeddings: http://nlp.stanford.edu/data/glove.840B.300d.zip

Unzip and save the text file to a subdirectory with the path: `GloVe/glove.840B.300d.txt`

### 4. Run the code:

For example, to run our method on the MNLI dataset represented as BERT contextual embeddings, assuming that `./MNLI` is the path to the dataset downloaded in the previous step and that your machine has a GPU:

```
python complexity.py \
    --dataset MNLI \
    --dataset_dir ./MNLI \
    --representation BERT \
    --sample_size 20000 \
    --run_number 1 \
    --gpu
```

The options for the `--dataset` flag are:
- `MNLI`
- `QNLI`
- `SNLI`
- `WNLI`

The path passed to the `--dataset_dir` flag must match the path you downloaded each dataset to in step 2.

The options for the `--representation` flag are:
- `bag-of-words`
- `glove`
- `BERT`
- `RoBERTa`
- `RoBERTa-large`
- `RoBERTa-large-mnli`
- `RoBERTa-large-qnli`
- `RoBERTa-large-snli`

The final three are only available if you fine-tuned RoBERTa-large in step 3. The `glove` option is available if you downloaded GloVe embeddings in step 4.

The `--gpu` flag is optional: include it when evaluating MLM contextual embeddings if your machine has a GPU.

The `--run_number` flag can be used to track replicates for the same combination of parameters.

To reproduce our main results in figure 8, you will need to run four replicates for every combination of `--dataset` and `--representation` for a fixed `--sample_size` of 20000:
{`MNLI`, `QNLI`, `SNLI`, `WNLI`} x {`bag-of-words`, `glove`, `BERT`, `RoBERTa-large`, `RoBERTa-large-mnli`, `RoBERTa-large-qnli`, `RoBERTa-large-snli`}

Each run of 20000 samples took several hours on our machine.

### 5. Plot the results:

```
python plot-results.py --sample_size 20000
```

Graphs are saved to a subdirectory called `graphs`. The `--sample_size` flag here tells the plotting script to plot all results with this sample size.

If you have not run all of the dataset/representation combinations we examine in the paper, you will see some empty bars in your graphs.




