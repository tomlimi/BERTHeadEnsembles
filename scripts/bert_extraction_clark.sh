#!/usr/bin/env bash

#DATA_FILE="/lnet/ms/projects/LSD/attention_tomasz/lsd/attention-analysis/naacl2019/BertAA/BertAA-ml-${LAN}-cased/4bert.json"
#BERT_MODEL="/lnet/ms/projects/bert/models/multilingual-base-cased"


DATA_FILE="/lnet/ms/projects/LSD/attention_tomasz/lsd/attention-analysis/naacl2019/BertAA/BertAA-PUD-${LAN}/4bert.json"
BERT_MODEL="/lnet/ms/projects/bert/models/multilingual-base-uncased/"

source /home/limisiewicz/general/bin/activate

python /lnet/ms/projects/LSD/attention_tomasz/lsd/attention-analysis/naacl2019/BertAA/attention-analysis/extract_attention.py --preprocessed-data-file $DATA_FILE --bert-dir $BERT_MODEL --max_sequence_length 256 #--cased
