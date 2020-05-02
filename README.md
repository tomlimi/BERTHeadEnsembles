# BERTHeadEnsembles

The repository contains code for [Universal Dependencies according to BERT: both more specific and more general](https://arxiv.org/abs/2004.14620)

## Universal Dependencies Modification

Our modification of the Universal Dependencies annotation is applied with UDApi. 
To install UDApi, follow the instruction from [UDApi](https://github.com/udapi/udapi-python). 
We have created our custom block that performs conllu modifications, to use it:

1. Clone the UDApi repository, 
2. Copy the file `attentionconverte.py` to `udapi-python/udapi/block/ud`
3. Follow the steps in _Install Udapi for developers_ for developers 
4. Run in a command line:

```
udapy read.Conllu files=<path-to-conllu> ud.AttentionConvert write.Conllu > <path-to-converted-conllu>
```

*Note that this step is optional. However, it is necessary to reproduce our results.*


## Extracting BERT Attention Maps
**The code and instruction for running BERT over text and extracting the resulting attention map were created by Kevin Clark and 
were adapted for this project. The original code is available at [Attention Analysis Clark et al.](https://github.com/clarkkev/attention-analysis)**

The input data should be a [JSON](https://www.json.org/) file containing a
list of dicts, each one corresponding to a single example to be passed into BERT. Each dict must contain exactly one of the following fields:
* `"text"`: A string.
* `"words"`: A list of strings. Needed if you want word-level rather than
token-level attention.
* `"tokens"`: A list of strings corresponding to BERT wordpiece tokenization.

If the present field is "tokens," the script expects [CLS]/[SEP] tokens
to be already added; otherwise it adds these tokens to the
beginning/end of the text automatically.
Note that if an example is longer than `max_sequence_length` tokens
after BERT wordpiece tokenization, attention maps will not be extracted for it.

Attention extraction is run with
```
python attention-analysis-clark-etal/extract_attention.py --preprocessed_data_file <path-to-your-data> --bert_dir <directory-containing-BERT-model> --max_sequence_length 256
```
The following optional arguments can also be added:
* `--max_sequence_length`: Maximum input sequence length after tokenization (default is 128).
* `--batch_size`: Batch size when running BERT over examples (default is 16).
* `--debug`: Use a tiny BERT model for fast debugging.
* `--cased`: Do not lowercase the input text.
* `--word_level`: Compute word-level instead of token-level attention (see Section 4.1 of the paper).

The list of attention matrices will be saved to  `<path-to-your-data>_attentions.npz`. The file will be referred to as `<path-to-attentions>` in the next steps.

Wordpiece tokenized sentences will be saved to `<path-to-your-data>_source.txt`.  The file will be referred to as `<path-to-wordpieces>` in the next steps.


## Head Ensemble Selection

Select syntactic head ensembles for each Universal Dependencies syntactic relation:

```
python3  head-ensembles/head_ensemble.py <attention-matrices> <bpe-tokenized-sentences> <path-to-conllu> -j <path-to-head-ensembles>
```

`<attention-matrices>` and `<bpe-tokenized-sentences>` were generated in the last step.

`<conllu-file>` is a path to conll file used for evaluation, that optionally was converted with UDApi before.

A dictionary is produced with syntactic labels as keys and head ensembles as values. Each head ensemble 
contains fields:
* ensemble: list of pairs [layer_index, head_index] of heads selected to the ensemble
* max_metric: metric result for the head ensemble on evaluation conllu (_Dependency accuracy_ by default)
* metric_history: metric result in each step of the selection process
* max_ensemble_size: the limit of the number of heads in an ensemble
* relation_label: the same as a dictionary key

If the argument `--json` is provided the dictionary is saved in `JSON` format.


Other arguments for the script:
* `--metric`: metric to optimize in head ensemble selection (currently only *DepAcc* is supported)
* `--num-heads`: the maximal size of each head ensemble (by default: 4)
* `--sentences`: indices of the sentences used for selection. 

## Dependency Tree Construction

Construct dependency trees from head ensembles selected in the last step and evaluate their UAS and LAS on conllu file.

```
python  head-ensembles/extract_trees.py <attention-matrices> <bpe-tokenized-sentences> <path-to-conllu> <path-to-head-ensmemble>
```

The results are printed to standard output.
We use different conllu file for head ensemble selection (*EuroParl* with UD modification) and dependency tree  (*PUD* w/o UD modifications)

Other arguments for the script:
* `--sentences`: indices of the sentences used for selection. 

## Citation

```
@misc{limisiewicz2020universal,
    title={Universal Dependencies according to BERT: both more specific and more general},
    author={Tomasz Limisiewicz and Rudolf Rosa and David Mare\v{c}ek},
    year={2020},
    eprint={2004.14620},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
