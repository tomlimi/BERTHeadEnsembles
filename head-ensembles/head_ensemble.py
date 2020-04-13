import argparse
import numpy as np
from collections import defaultdict

from dependency import Dependency
from atttention_wrapper import AttentionWrapper
from metrics import DepAcc


HEADS_TO_CHECK = 25


class HeadEnsemble():
    MAX_ENSEMBLE_SIZE = 4
    def __init__(self):
        self.ensemble = list()
        self.max_metric = 0.
        self.metric_history = list()

    def consider_candidate(self, candidate, metric, attn_wrapper):

        candidate_lid, candidate_hid = candidate
        if len(self.ensemble) < self.MAX_ENSEMBLE_SIZE:
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
    ap.add_argument("attentions", required=True, help="NPZ file with attentions")
    ap.add_argument("tokens", required=True, help="Labels (tokens) separated by spaces")
    ap.add_argument("conll", help="Conll file for head selection.")

    ap.add_argument("-m", "--metric", help="Metric  used ")\

    # other arguments

    ap.add_argument("-s", "--sentences", nargs='*', type=int, default=None,
                    help="Only use the specified sentences; 0-based")

    args = ap.parse_args()

    dependency_tree = Dependency(args.conll)
    bert_attns = AttentionWrapper(args.attentions, dependency_tree)

    metric = None
    head_ensembles = dict(HeadEnsemble)
    for relation_label in dependency_tree.label_map.keys:
        if args.metric.lower() == "depacc":
            metric = DepAcc(dependency_tree.relations, relation_label)
        else:
            raise ValueError("Unknown metric! Available metrics: DepAcc")
        print(f"Finding Head Ensemble for label: {relation_label}")

        metric_grid = bert_attns.calc_metric_all_heads(metric)
        heads_ids = np.argsort(metric_grid, axis=None)[-HEADS_TO_CHECK:][::-1]

        for candidate in np.unravel_index(heads_ids, (12,12)):
            head_ensembles[relation_label].consider_candidate(candidate)