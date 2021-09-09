## data-label-alignment

Code for "Comparing Text Representations: A Theory-Driven Approach" (EMNLP 2021).

These are instructions for reproducing the paper's main results on natural language inference datasets.

**Coming soon:** code to easily run data-label alignment on your own datasets!

### 1. Install requirements.

This code requires Python 3 (version >= 3.6). Use pip to install requirements:

```
pip install -r requirements.txt
```

### 2. Download NLI datasets:

- MNLI (298.3 MB): https://dl.fbaipublicfiles.com/glue/data/MNLI.zip 
- QNLI (10.1 MB): https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip
- SNLI (90.2 MB): https://nlp.stanford.edu/projects/snli/snli_1.0.zip 
- WNLI (28.3 KB): https://dl.fbaipublicfiles.com/glue/data/WNLI.zip

We recommend saving these inside the current directory.

### 3. *Optional:* fine-tune RoBERTa-large on subsets of MNLI, QNLI, SNLI.

Fine-tuning will allow you to use our method to evaluate fine-tuned MLM contextual embeddings. For example:

```
python finetune.py \
    --dataset QNLI \
    --dataset_dir ./QNLI \
    --lr 2e-5
```

The options for the `--dataset` flag are `MNLI`, `QNLI`, and `SNLI`. The path passed to the `--dataset_dir` flag must match the path you downloaded data to in step 3. The `--lr` flag specifies the initial learning rate.

You can skip this step if you wish to only evaluate baseline non-contextual and pre-trained MLM contextual embeddings.

### 4. *Optional:* download GloVe embeddings.

Download pre-trained GloVe embeddings: http://nlp.stanford.edu/data/glove.840B.300d.zip

Unzip and save the text file to a subdirectory with the path: `GloVe/glove.840B.300d.txt`

### 5. Run the code:

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

### 6. Plot the results:

```
python plot-results.py --sample_size 20000
```

Graphs are saved to a subdirectory called `graphs`. The `--sample_size` flag here tells the plotting script to plot all results with this sample size.

If you have not run all of the dataset/representation combinations we examine in the paper, you will see some empty bars in your graphs.




