#!/usr/bin/env bash

FILE=/ha/home/limisiewicz/attention_my/BertAA/BertAA-ml-${LAN}

OUTPUT_DIR=/lnet/ms/projects/LSD/attention-analysis/acl2020/graph-extraction/experiments
UAS_SCRIPT=/lnet/ms/projects/LSD/attention-analysis/acl2020/scripts/attention_uas_multihead_max_heads.py
CONVCONLLU=/lnet/ms/projects/LSD/attention-analysis/acl2020/graph-extraction/${LAN}dev.conllu
TESTDIR=/ha/home/limisiewicz/attention_my/BertAA/BertAA-PUD-${LAN}
TESTCONLLU=/lnet/ms/projects/LSD/attention-analysis/acl2020/graph-extraction/${LAN}_pud-ud-test.conllu

source /home/limisiewicz/general/bin/activate

echo "language ${LAN}"
echo "number heads ${NHEADS}"
echo "number sentences ${NSENTS}"
echo "randomization"


b=$(basename $FILE)
if [ ! -d  "$OUTPUT_DIR/$b/selheads${NSENTS}" ]
then
  mkdir -p "$OUTPUT_DIR/$b/selheads${NSENTS}"
fi

i=0
python $UAS_SCRIPT -a "${FILE}/attentions.npz" -ta "$TESTDIR/attentions.npz" -t "$FILE/source.txt"  -tt "$TESTDIR/source.txt"\
  -j "$OUTPUT_DIR/$b/selheads${NSENTS}/run${i}" -c $CONVCONLLU -tc $TESTCONLLU -e --numheads $NHEADS -s $NSENTS -r

#for i in {0..20}
#do
#  python $UAS_SCRIPT -a "${FILE}/attentions.npz" -ta "$TESTDIR/attentions.npz" -t "$FILE/source.txt"  -tt "$TESTDIR/source.txt"\
#  -j "$OUTPUT_DIR/$b/selheads${NSENTS}/run${i}" -c $CONVCONLLU -tc $TESTCONLLU -e -tn --numheads $NHEADS -s $NSENTS -r
#done
