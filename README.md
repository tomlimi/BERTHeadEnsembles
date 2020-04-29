# BERTHeadEnsembles


## UD modification

## Extracting BERT Attention Maps
**The code and instruction for running BERT over text and extracting the resulting attention map were created by Kevin Clark and 
was modified in a small extent for this project. The original code is available at https://github.com/clarkkev/attention-analysis**

The input data should be a [JSON](https://www.json.org/) file containing a
list of dicts, each one corresponding to a single example to be passed in
to BERT. Each dict must contain exactly one of the following fields:
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
python3 attention-analysis-clarks-etal/extract_attention.py --preprocessed_data_file <path-to-your-data> --bert_dir <directory-containing-BERT-model> --max_sequence_length 256
```
The following optional arguments can also be added:
* `--max_sequence_length`: Maximum input sequence length after tokenization (default is 128).
* `--batch_size`: Batch size when running BERT over examples (default is 16).
* `--debug`: Use a tiny BERT model for fast debugging.
* `--cased`: Do not lowercase the input text.
* `--word_level`: Compute word-level instead of token-level attention (see Section 4.1 of the paper).

The list of attention matrices (without rows and columns corresponding to [CLS], [EOS]) will be saved to  `<path-to-your-data>_attentions.npz`. The file will be referred as `<path-to-attentions>` in the next steps.
Wordpiece tokenized sentences will be saved to `<path-to-your-data>_source.txt`.  The file will be referred as `<path-to-wordpieces>` in the next steps.


## Head extraction

To find syntactic head ensmbles please run:

```
python3  head-ensembles/head_ensemble.py <attention-matrices> <bpe-tokenized-sentences> -j <path-to-head-ensembles>
```


## Dependency Tree construction

```
python3  <attention-matrices> <bpe-tokenized-sentences> -j <head-ensmemble-json>
```


