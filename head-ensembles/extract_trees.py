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
	
	ap.add_argument("-s", "--sentences", nargs='*', type=int, default=None,
	                help="Only use the specified sentences; 0-based")
	
	args = ap.parse_args()
	
	dependency_tree = Dependency(args.conll, args.tokens)
	
	head_ensembles = dict()
	ensembles_d2p = dict()
	ensembles_p2d = dict()
	depacc_d2p = dict()
	depacc_p2d = dict()
	
	with open(args.json, 'r') as inj:
		head_ensembles = json.load(inj)
	
	considered_relations = ('adj-modifier', 'adv-modifier', 'auxiliary', 'compound', 'conjunct', 'determiner',
							'noun-modifier', 'num-modifier', 'object', 'subject', 'cc', 'case', 'mark')

	for relation in considered_relations:
		ensembles_d2p[relation] = head_ensembles[relation + '-d2p']['ensemble']
		depacc_d2p[relation] = head_ensembles[relation + '-d2p']['max_metric']
		ensembles_p2d[relation] = head_ensembles[relation + '-p2d']['ensemble']
		depacc_p2d[relation] = head_ensembles[relation + '-p2d']['max_metric']
		
	bert_attns = AttentionWrapper(args.attentions, dependency_tree.wordpieces2tokens, args.sentences)
	extracted_unlabeled, extracted_labeled = bert_attns.extract_trees(ensembles_d2p, ensembles_p2d, depacc_d2p, depacc_p2d, dependency_tree.roots)
	
	uas_m = UAS(dependency_tree)
	las_m = LAS(dependency_tree)
	
	uas_res = uas_m(bert_attns.sentence_idcs, extracted_unlabeled).result()
	las_res = las_m(bert_attns.sentence_idcs, extracted_labeled).result()

	print(f"UAS result for extracted tree: {uas_res}, LAS: {las_res}")