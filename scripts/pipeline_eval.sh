#!/bin/bash
source /home/limisiewicz/.virtualenvs/headensemble/bin/activate

export PATH="/home/limisiewicz/udapi-python/bin:$PATH"
export PYTHONPATH="/home/limisiewicz/udapi-python/:$PYTHONPATH"

BERTDIR='/net/projects/bert/models/multilingual-base-uncased/'

PREFIXPROCESS=$1
PREFIXEVAL=$2
RESOURCESDIR='../resources'
RESULTDIR='../results'


PROCESSFILE=$RESOURCESDIR/$PREFIXPROCESS
EVALFILE=$RESOURCESDIR/$PREFIXEVAL
RESULTFILE=$RESULTDIR/$PREFIXEVAL

# convert conllus
udapy read.Conllu files="${PROCESSFILE}.conllu" ud.AttentionConvert write.Conllu > "${PROCESSFILE}-conv.conllu"
#
udapy read.Conllu files="${EVALFILE}.conllu" ud.AttentionConvert write.Conllu > "${EVALFILE}-conv.conllu"

# select heads
python3  ../head-ensembles/head_ensemble.py "${PROCESSFILE}_attentions.npz" "${PROCESSFILE}_source.txt" "${PROCESSFILE}-conv.conllu" -j "${PROCESSFILE}_head-ensembles.json" --report-result "${RESULTDIR}/${PREFIXPROCESS}.dep_acc" --offsets "${PROCESSFILE}_offsets.json"

#evaluate
python3  ../head-ensembles/head_ensemble.py "${EVALFILE}_attentions.npz" "${EVALFILE}_source.txt" "${EVALFILE}-conv.conllu" -j "${PROCESSFILE}_head-ensembles.json" --report-result "${RESULTFILE}.dep_acc" -e --offsets "${PROCESSFILE}_offsets.json"
python3 ../head-ensembles/extract_trees.py "${EVALFILE}_attentions.npz" "${EVALFILE}_source.txt" "${EVALFILE}.conllu" "${PROCESSFILE}_head-ensembles.json" --report-result "${RESULTFILE}.trees"

# optional cleanup
rm "${PROCESSFILE}.json"
rm "${PROCESSFILE}-conv.conllu"
#rm "${PROCESSFILE}_attentions.npz"
#rm "${PROCESSFILE}_source.txt"
#
rm "${EVALFILE}.json"
rm "${EVALFILE}-conv.conllu"
#rm "${EVALFILE}_attentions.npz"
#rm "${EVALFILE}_source.txt"
