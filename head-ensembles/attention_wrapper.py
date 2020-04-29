import numpy as np
import sys
from itertools import chain
from tqdm import tqdm
from multiprocessing import Pool

import networkx as nx
from networkx.algorithms import tree

class AttentionWrapper:
    # Those values are used in all the experiments. Parameters could be superfluous.
    MAX_LEN = 1000  # maximum number of tokens in the sentence
    WITH_EOS = True  # whether attention matrix contain EOS token.
    WITH_CLS = False # whether attention matrix contain CLS token.
    NO_SOFTMAX = False  # whether to conduct softmax on loaded attention matrices. Should be True for endev.

    def __init__(self, attention_file, tokens_grouped, selected_sentences=None):

        # loads all the attention matrices and tokens
        attention_loaded = np.load(attention_file)
        self.layer_count = attention_loaded['arr_0'].shape[0]
        self.head_count = attention_loaded['arr_0'].shape[1]

        self.matrices = list()
        self.tokens_grouped = tokens_grouped
        self.sentence_idcs = selected_sentences or list(range(len(attention_loaded.files)))

        self.preprocess_matrices(attention_loaded)

    def calc_metric_grid(self, metric):
        metric_res = np.zeros((self.layer_count, self.head_count))
        for l in range(self.layer_count):
            for h in range(self.head_count):
                metric.reset_state()
                metric(self.sentence_idcs, [np.squeeze(sent_matrices[l,h,:,:]) for sent_matrices in self.matrices])
                metric_res[l,h] = metric.result()
        return metric_res

    def calc_metric_ensemble(self, metric, layer_idx, head_idx):
        metric.reset_state()
        metric(self.sentence_idcs,
               [sent_matrices[layer_idx, head_idx, :,:].mean(axis=0) for sent_matrices in self.matrices])
        return metric.result()

    # TODO: remove tree extraction logic from wrapper
    def extract_trees(self, relation_heads_d2p, relation_heads_p2d, weights_d2p, weights_p2d, roots):
        extracted_unlabeled = list()
        extracted_labeled = list()
        for idx, sent_idx in tqdm(enumerate(self.sentence_idcs), desc= 'Extracting trees from matrices'):
            root = roots[sent_idx]
            dependency_graph = nx.MultiDiGraph()
            dependency_graph.add_nodes_from(range(len(self.tokens_grouped[sent_idx])))

            edge2relation_label = dict()
            for relation in relation_heads_d2p.keys():

                layer_idx_d2p, head_idx_d2p = zip(*relation_heads_d2p[relation])
                ensemble_matrix_d2p = self.matrices[idx][layer_idx_d2p, head_idx_d2p,:,:].mean(axis=0).transpose()
                ensemble_matrix_d2p[:, root] = 0.001
                np.fill_diagonal(ensemble_matrix_d2p, 0.001)
                ensemble_matrix_d2p = np.clip(ensemble_matrix_d2p, 0.001, 0.999)

                layer_idx_p2d, head_idx_p2d = zip(*relation_heads_p2d[relation])
                ensemble_matrix_p2d = self.matrices[idx][layer_idx_p2d, head_idx_p2d, :, :].mean(axis=0)
                ensemble_matrix_p2d[:,root] = 0.001
                np.fill_diagonal(ensemble_matrix_p2d, 0.001)
                ensemble_matrix_p2d = np.clip(ensemble_matrix_p2d, 0.001, 0.999)

                weight_p2d = weights_p2d[relation] ** 5
                weight_d2p = weights_d2p[relation] ** 5
                ensemble_matrix = (weight_d2p * np.log(ensemble_matrix_d2p) + weight_p2d * np.log(ensemble_matrix_p2d)) / (weight_d2p + weight_p2d)

                ensemble_graph = nx.from_numpy_matrix(ensemble_matrix, create_using=nx.DiGraph)

                # Unfortunately this is necessary, because netwokx multigraph loses information about edges
                for u, v, d in ensemble_graph.edges(data=True):
                    edge2relation_label[(u, v, d['weight'])] = relation

                dependency_graph.add_edges_from(ensemble_graph.edges(data=True), label=relation)

            dependency_aborescene = tree.branchings.maximum_spanning_arborescence(dependency_graph)

            extracted_unlabeled.append([(dep, parent) for parent, dep in dependency_aborescene.edges(data=False)] + [(root, -1)])
            extracted_labeled.append([(dep, parent, edge2relation_label[(parent, dep, edge_data['weight'])])
                                      for parent, dep, edge_data in dependency_aborescene.edges(data=True)] + [(root, -1, 'root')])

        return extracted_unlabeled, extracted_labeled

    def __getitem__(self, idx):
        return self.sentence_idcs[idx], self.matrices[idx]

    def check_wordpieces(self, item, attention_loaded):
        matrix_id = 'arr_' + str(item)
        attention_rank = attention_loaded[matrix_id].shape[2] - int(self.WITH_EOS) - int(self.WITH_CLS)
        item_wordpieces_grouped = self.tokens_grouped[item]
        if item_wordpieces_grouped is None:
            print('Token mismatch sentence skipped', item, file=sys.stderr)
            return False

        item_wordpieces = list(chain.from_iterable(item_wordpieces_grouped))
        # check maxlen
        if not len(item_wordpieces_grouped) <= self.MAX_LEN:
            print('Too long sentence, skipped', item, file=sys.stderr)
            return False
        # NOTE sentences truncated to 64 tokens
        if len(item_wordpieces) != attention_rank:
            print('Too long sentence, skipped', item, file=sys.stderr)
            return False
        return True

    def aggregate_wordpiece_matrices(self, attention_matrices, tokens_grouped):
        # this functions connects wordpieces and aggregates their attention.
        midres_matrices = np.zeros((self.layer_count, self.head_count, len(tokens_grouped), attention_matrices.shape[3]))

        for tok_id, wp_ids in enumerate(tokens_grouped):
            midres_matrices[:,:,tok_id, :] = np.mean(attention_matrices[:,:,wp_ids, :], axis=2)

        res_matrices= np.zeros((self.layer_count, self.head_count, len(tokens_grouped), len(tokens_grouped)))

        for tok_id, wp_ids in enumerate(tokens_grouped):
            res_matrices[:,:,:, tok_id] = np.sum(midres_matrices[:, :, :, wp_ids], axis=3)

        return res_matrices

    def preprocess_matrices(self, attention_loaded):
        for sent_idx in tqdm(self.sentence_idcs[:], desc="Preprocessing attention for sentences"):
            if not self.check_wordpieces(sent_idx, attention_loaded):
                self.sentence_idcs.remove(sent_idx)
                continue

            matrices_id = 'arr_' + str(sent_idx)
            sent_matrices = np.array(attention_loaded[matrices_id])
            if self.WITH_EOS:
                sent_matrices = sent_matrices[:,:,:-1, :-1]
            if self.WITH_CLS:
                sent_matrices = sent_matrices[:, :, 1:, 1:]
            # the max trick -- for each row subtract its max
            # from all of its components to get the values into (-inf, 0]
            if not self.NO_SOFTMAX:
                sent_matrices = sent_matrices - np.max(sent_matrices, axis=3, keepdims=True)
                exp_matrix = np.exp(sent_matrices)
                sent_matrices = exp_matrix / np.sum(exp_matrix, axis=3, keepdims=True)
            else:
                sent_matrices = sent_matrices / np.sum(sent_matrices, axis=3, keepdims=True)
            sent_matrices = self.aggregate_wordpiece_matrices(sent_matrices, self.tokens_grouped[sent_idx])
            self.matrices.append(sent_matrices)
