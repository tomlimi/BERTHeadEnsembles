import numpy as np
import argparse
import json


from dependency import Dependency
from metrics import UAS
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
	selected_head_ensembles = dict()
	selected_directions = dict()
	
	with open(args.json, 'r') as inj:
		head_ensembles = json.load(inj)
	
	for relation in set(dependency_tree.label_map.values()):
		if head_ensembles[relation + '-p2d']['max_metric'] > head_ensembles[relation + '-d2p']['max_metric']:
			if head_ensembles[relation + '-p2d']['max_metric'] > DEPACC_THRESHOLD:
				selected_head_ensembles[relation] = head_ensembles[relation + '-p2d']['ensemble']
				print(selected_head_ensembles[relation] )
				selected_directions[relation] = 'p2d'
		else:
			if head_ensembles[relation + '-d2p']['max_metric'] > DEPACC_THRESHOLD:
				selected_head_ensembles[relation] = head_ensembles[relation + '-d2p']['ensemble']
				selected_directions[relation] = 'd2p'
		
	bert_attns = AttentionWrapper(args.attentions, dependency_tree.wordpieces2tokens, args.sentences)
	extracted_labeled, extracted_unlabeled = bert_attns.extract_trees(selected_head_ensembles, selected_directions, dependency_tree.roots)
	
	uas_m = UAS(dependency_tree.unlabeled_relations)
	
	uas_res = uas_m.calculate(bert_attns.sentence_idcs, extracted_unlabeled)

	print(f"UAS result for extracted tree: {uas_res}")