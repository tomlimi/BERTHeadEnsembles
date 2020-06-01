#!/bin/bash
source /home/limisiewicz/.virtualenvs/headensemble/bin/activate

PREFIXPROCESS=$1
RESOURCESDIR='../resources'
BERTDIR='/net/projects/bert/models/multilingual-base-uncased/'

PROCESSFILE=$RESOURCESDIR/$PREFIXPROCESS

#python3 ../head-ensembles/conllu2json.py "${PROCESSFILE}.conllu" "${PROCESSFILE}.json"
#python3 ../attention-analysis-clark-etal/extract_attention.py --preprocessed-data-file "${PROCESSFILE}.json" --bert-dir $BERTDIR --max-sequence-length 256

# convert conll
udapy read.Conllu files="${PROCESSFILE}.conllu" ud.AttentionConvert write.Conllu > "${PROCESSFILE}-conv.conllu"

# select heads
python3  ../head-ensembles/head_ensemble.py "${PROCESSFILE}_attentions.npz" "${PROCESSFILE}_source.txt" "${PROCESSFILE}-conv.conllu" -j "${PROCESSFILE}_head-ensembles.json"

#extract trees
python3 ../head-ensembles/extract_trees.py "${PROCESSFILE}_attentions.npz" "${PROCESSFILE}_source.txt" "${PROCESSFILE}.conllu" "${PROCESSFILE}_head-ensembles.json"

# optional cleanup
#rm "${PROCESSFILE}.json"
#rm "${PROCESSFILE}-conv.conllu"
#rm "${PROCESSFILE}_attentions.npz"
#rm "${PROCESSFILE}_source.txt"