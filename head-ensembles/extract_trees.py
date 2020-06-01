#!/usr/bin/env python3

import numpy as np
import argparse
import json

from dependency import Dependency
from metrics import UAS, LAS
from attention_wrapper import AttentionWrapper

DEPACC_THRESHOLD = 0.6


def print_tikz(prediction_edges, sent_idcs, dependency , out_tikz_file):
	''' Turns edge sets on word (nodes) into tikz dependency LaTeX. '''
	uas_m = UAS(dependency)
	with open(out_tikz_file, 'w') as fout:
		for sent_preds, sid in zip(prediction_edges, sent_idcs):
			tokens = dependency.tokens[sid]
			uas_m.reset_state()
			uas_m([sid],[sent_preds])
			if len(tokens) < 10 and uas_m.result() > 0.6:
			
				sent_golds = dependency.unlabeled_relations[sid]
			
				string = """\\begin{dependency}[hide label, edge unit distance=.5ex]
		  \\begin{deptext}[column sep=0.05cm]
		  """
				string += "\\& ".join([x.replace('$', '\$').replace('&', '+') for x in tokens]) + " \\\\" + '\n'
				string += "\\end{deptext}" + '\n'
				for i_index, j_index in sent_golds:
					if i_index >= 0 and j_index >= 0:
						string += '\\depedge{{{}}}{{{}}}{{{}}}\n'.format(i_index + 1, j_index + 1, '.')
				for i_index, j_index in sent_preds:
					if i_index >= 0 and j_index >= 0:
						string += '\\depedge[edge style={{blue!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n'.format(i_index +1,
					                                                                                     j_index +1, '.')
				string += '\\end{dependency}\n'
				fout.write('\n\n')
				fout.write(string)


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
		
		print_tikz(extracted_unlabeled, bert_attns.sentence_idcs, dependency, args.report_result+".tikz")
		
		
	else:
		print(f"UAS result for extracted tree: {uas_res}, LAS: {las_res}")