import argparse
import numpy as np
import json
from tqdm import tqdm

from dependency import Dependency
from attention_wrapper import AttentionWrapper
from metrics import DepAcc


HEADS_TO_CHECK = 25


class HeadEnsemble():
    MAX_ENSEMBLE_SIZE = 4
    def __init__(self, relation_label):
        self.ensemble = list()
        self.max_metric = 0.
        self.metric_history = list()

        self.relation_label = relation_label

    def consider_candidate(self, candidate, metric, attn_wrapper):

        candidate_lid, candidate_hid = candidate
        if not self.ensemble:
            self.max_metric = attn_wrapper.calc_metric_ensemble(metric, [candidate_lid], [candidate_hid])
            self.ensemble.append(candidate)
        elif len(self.ensemble) < self.MAX_ENSEMBLE_SIZE:
            ensemble_lids, ensemble_hids = map(list, zip(*self.ensemble))
            candidate_metric = attn_wrapper.calc_metric_ensemble(metric, ensemble_lids + [candidate_lid],
                                                             ensemble_hids + [candidate_hid])
            if candidate_metric > self.max_metric:
                self.max_metric = candidate_metric
                self.ensemble.append(candidate)
        else:
            max_candidate_metric = 0.
            opt_substitute_idx = None
            for substitute_idx in range(self.MAX_ENSEMBLE_SIZE):
                ensemble_lids, ensemble_hids = map(list, zip(*self.ensemble))
                ensemble_lids[substitute_idx] = candidate_lid
                ensemble_hids[substitute_idx] = candidate_hid
                candidate_metric = attn_wrapper.calc_metric_ensemble(metric, ensemble_lids, ensemble_hids)
                if candidate_metric > self.max_metric and candidate_metric > max_candidate_metric:
                    max_candidate_metric = candidate_metric
                    opt_substitute_idx = substitute_idx

            if opt_substitute_idx is not None:
                self.ensemble[opt_substitute_idx] = candidate
                self.max_metric = max_candidate_metric
        self.metric_history.append(self.max_metric)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("attentions", type=str, help="NPZ file with attentions")
    ap.add_argument("tokens", type=str, help="Labels (tokens) separated by spaces")
    ap.add_argument("conll", type=str, help="Conll file for head selection.")

    ap.add_argument("-m", "--metric", type=str, default="DepAcc", help="Metric  used ")
    ap.add_argument("-j", "--json", type=str, help="Output json with the heads")
    # other arguments

    ap.add_argument("-s", "--sentences", nargs='*', type=int, default=None,
                    help="Only use the specified sentences; 0-based")

    args = ap.parse_args()

    dependency_tree = Dependency(args.conll, args.tokens)
    bert_attns = AttentionWrapper(args.attentions, dependency_tree.wordpieces2tokens)

    metric = None
    head_ensembles = dict()
    for direction in ['d2p', 'p2d']:
        for relation_label in dependency_tree.label_map.keys():
            if args.metric.lower() == "depacc":
                metric = DepAcc(dependency_tree.relations, relation_label, dependent2parent=(direction=='d2p'))
            else:
                raise ValueError("Unknown metric! Available metrics: DepAcc")
            relation_label_directional = relation_label + '-' + direction
            head_ensembles[relation_label] = HeadEnsemble(relation_label_directional)
            print(f"Calculating metric for each head. Relation label: {relation_label_directional}")
            metric_grid = bert_attns.calc_metric_single(metric)
            heads_idcs = np.argsort(metric_grid, axis=None)[-HEADS_TO_CHECK:][::-1]
            for candidate_id in tqdm(heads_idcs, desc=f"Candidates for ensemble!"):
                candidate = np.unravel_index(candidate_id, metric_grid.shape)
                head_ensembles[relation_label_directional].consider_candidate(candidate, metric, bert_attns)

    if args.json:
        with open(args.json, 'w') as outj:
            json.dump(head_ensembles, fp=outj)