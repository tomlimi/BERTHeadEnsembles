#!/bin/bash
BERTDIR=$3

PREFIXPROCESS=$1
PREFIXEVAL=$2
RESOURCESDIR='../resources'
RESULTDIR='../results'


PROCESSFILE=$RESOURCESDIR/$PREFIXPROCESS
EVALFILE=$RESOURCESDIR/$PREFIXEVAL
RESULTFILE=$RESULTDIR/$PREFIXEVAL


# prepare attention matrices
python3 ../head-ensembles/conllu2json.py "${PROCESSFILE}.conllu" "${PROCESSFILE}.json"
python3 ../attention-analysis-clark-etal/extract_attention.py --preprocessed-data-file "${PROCESSFILE}.json" --bert-dir $BERTDIR --max-sequence-length 256

python3 ../head-ensembles/conllu2json.py "${EVALFILE}.conllu" "${EVALFILE}.json"
python3 ../attention-analysis-clark-etal/extract_attention.py --preprocessed-data-file "${EVALFILE}.json" --bert-dir $BERTDIR --max-sequence-length 256

# convert conllus
udapy read.Conllu files="${PROCESSFILE}.conllu" ud.AttentionConvert write.Conllu > "${PROCESSFILE}-conv.conllu"

udapy read.Conllu files="${EVALFILE}.conllu" ud.AttentionConvert write.Conllu > "${EVALFILE}-conv.conllu"

# select heads
python3  ../head-ensembles/head_ensemble.py "${PROCESSFILE}_attentions.npz" "${PROCESSFILE}_source.txt" "${PROCESSFILE}-conv.conllu" -j "${PROCESSFILE}_head-ensembles.json" --report-result "${RESULTDIR}/${PREFIXPROCESS}.dep_acc"

#evaluate
python3  ../head-ensembles/head_ensemble.py "${EVALFILE}_attentions.npz" "${EVALFILE}_source.txt" "${EVALFILE}-conv.conllu" -j "${PROCESSFILE}_head-ensembles.json" --report-result "${RESULTFILE}.dep_acc" -e
python3 ../head-ensembles/extract_trees.py "${EVALFILE}_attentions.npz" "${EVALFILE}_source.txt" "${EVALFILE}.conllu" "${PROCESSFILE}_head-ensembles.json" --report-result "${RESULTFILE}.trees"

# optional cleanup
rm "${PROCESSFILE}.json"
rm "${PROCESSFILE}-conv.conllu"
rm "${PROCESSFILE}_attentions.npz"
rm "${PROCESSFILE}_source.txt"

rm "${EVALFILE}.json"
rm "${EVALFILE}-conv.conllu"
rm "${EVALFILE}_attentions.npz"
rm "${EVALFILE}_source.txt"