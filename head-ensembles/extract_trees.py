#!/usr/bin/env python3

import numpy as np
import argparse
import json

from dependency import Dependency
from metrics import UAS, LAS
from attention_wrapper import AttentionWrapper

DEPACC_THRESHOLD = 0.6

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("attentions", type=str, help="NPZ file with attentions")
	ap.add_argument("tokens", type=str, help="Labels (tokens) separated by spaces")
	ap.add_argument("conll", type=str, help="Conll file for head selection.")
	ap.add_argument("json", type=str, help="Json file with head ensemble")
	
	# other arguments
	ap.add_argument("--report-result", type=str, default=None, help="File where to save the results.")
	ap.add_argument("-s", "--sentences", nargs='*', type=int, default=None,
	                help="Only use the specified sentences; 0-based")
	
	args = ap.parse_args()
	
	dependency = Dependency(args.conll, args.tokens)
	
	head_ensembles = dict()
	ensembles_d2p = dict()
	ensembles_p2d = dict()
	depacc_d2p = dict()
	depacc_p2d = dict()
	
	with open(args.json, 'r') as inj:
		head_ensembles = json.load(inj)
	
	# considered_relations = (Dependency.LABEL_ALL,)
	
	considered_relations = ('adj-modifier', 'adv-modifier', 'auxiliary', 'compound', 'conjunct', 'determiner',
							'noun-modifier', 'num-modifier', 'object', 'other', 'subject', 'cc', 'case', 'mark')

	for relation in considered_relations:
		ensembles_d2p[relation] = head_ensembles[relation + '-d2p']['ensemble']
		depacc_d2p[relation] = head_ensembles[relation + '-d2p']['max_metric']
		ensembles_p2d[relation] = head_ensembles[relation + '-p2d']['ensemble']
		depacc_p2d[relation] = head_ensembles[relation + '-p2d']['max_metric']
		
	bert_attns = AttentionWrapper(args.attentions, dependency.wordpieces2tokens, args.sentences)
	extracted_unlabeled, extracted_labeled = bert_attns.extract_trees(ensembles_d2p, ensembles_p2d, depacc_d2p, depacc_p2d, dependency.roots)
	
	uas_m = UAS(dependency)
	uas_m(bert_attns.sentence_idcs, extracted_unlabeled)
	uas_res = uas_m.result()

	las_m = LAS(dependency)
	las_m(bert_attns.sentence_idcs, extracted_labeled)
	las_res = las_m.result()
	
	if args.report_result:
		with open(args.report_result, 'w') as res_file:
			res_file.write(f"UAS: {uas_res}\n")
			res_file.write(f"LAS: {las_res}\n")
	else:
		print(f"UAS result for extracted tree: {uas_res}, LAS: {las_res}")