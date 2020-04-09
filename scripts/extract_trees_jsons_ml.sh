#!/usr/bin/env bash

f=/ha/home/limisiewicz/attention_my/lsd/attention-analysis/naacl2019/BertAA/BertAA-PUD-${LAN}/


EXTRAXT_SCRIPT=/net/projects/LSD/attention_tomasz/lsd/attention-analysis/acl2020/scripts/extract_trees_em.py

SELHEADS=/net/projects/LSD/attention_tomasz/lsd/attention-analysis/acl2020/graph-extraction/experiments/BertAA-ml-${LAN}/selheads${NSENTS}
CONLLU=/net/projects/LSD/attention_tomasz/lsd/attention-analysis/acl2020/graph-extraction/PUD-${LAN}/${LAN}_pud-ud-test.conllu



echo "language: ${LAN}"
echo "Number of sentences: ${NSENTS}"
echo "Discarding factor: ${DF}"

source /home/limisiewicz/general/bin/activate

export PATH=/home/limisiewicz/udapi-python/bin:$PATH
export PYTHONPATH=/home/limisiewicz/udapi-python/:$PYTHONPATH


for json in "$SELHEADS"/*heads2.json; do
  s=$(uuidgen)

  CONLLUPRED=/net/projects/LSD/attention_tomasz/lsd/attention-analysis/naacl2019/graph-extraction/em_conllu/${LAN}pud-pred-${NSENTS}-${s}.conllu
  CONLLUGOLD=/net/projects/LSD/attention_tomasz/lsd/attention-analysis/naacl2019/graph-extraction/em_conllu/${LAN}pud-gold-${NSENTS}-${s}.conllu
  echo "run id: ${s}"
  python $EXTRAXT_SCRIPT -a "$f/attentions.npz" -t "$f/source.txt" -e -c $CONLLU -o $CONLLUPRED -g $CONLLUGOLD -f -s $s\
   -d ${DF} -j ${json}
  udapy read.Conllu files=$CONLLUGOLD zone=gold read.Conllu zone=pred files=$CONLLUPRED  eval.Conll18

done
#do
#  python $EXTRAXT_SCRIPT -a "$f/attentions.npz" -t "$f/source.txt" -e -c $CONLLU -o $CONLLUPRED -g $CONLLUGOLD -s $s -d ${DF}
#  udapy read.Conllu files=$CONLLUGOLD zone=gold read.Conllu zone=pred files=$CONLLUPRED  eval.Conll18
#done
