import numpy as np
from abc import abstractmethod
from collections import defaultdict


class Metric:

    def __init__(self, dependency, *args, **kwargs):
        self.dependency = dependency

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def update_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def result(self):
        pass


class DepAcc(Metric):

    def __init__(self, dependency, relation_label, dependent2parent=True):

        self.relation_label = relation_label
        self.dependent2parent = dependent2parent

        self.retrieved = 0
        self.total = 0
        super().__init__(dependency)

    def __call__(self, sent_idcs, matrices):
        for sent_id, matrix in zip(sent_idcs, matrices):
            self.update_state(sent_id, matrix)

    def reset_state(self):
        self.retrieved = 0
        self.total = 0

    def update_state(self, sent_id, matrix):
        if matrix is not None:
            np.fill_diagonal(matrix, 0.)
            max_row = matrix.argmax(axis=1)

            rel_pairs = self.dependency.relations[self.relation_label][sent_id]
            if not self.dependent2parent:
                rel_pairs = list(map(tuple, map(reversed, rel_pairs)))
            self.retrieved += sum([max_row[attending] == attended for attending, attended in rel_pairs])
            self.total += len(rel_pairs)

    def result(self):
        return self.retrieved / self.total


class UAS(Metric):

    def __init__(self, dependency):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0
        super().__init__(dependency)

    def __call__(self, sent_idcs, predicted_relations):
        for sent_id, sent_predicted_relations in zip(sent_idcs, predicted_relations):
            self.update_state(sent_id, sent_predicted_relations)

    def reset_state(self):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0

    def update_state(self, sent_id, sent_predicted_relations):
        if sent_predicted_relations is not None:
            rel_pairs = self.dependency.unlabeled_relations[sent_id]
            self.all_gold += len(rel_pairs)
            self.all_predicted += len(sent_predicted_relations)
            self.all_correct += len(set(rel_pairs).intersection(set(sent_predicted_relations)))

    def result(self):
        if not self.all_correct:
            return 0.
        return 2. / (self.all_predicted / self.all_correct + self.all_gold / self.all_correct)


class LAS(Metric):
    def __init__(self, dependency):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0
        super().__init__(dependency)

    def __call__(self, sent_idcs, predicted_relations):
        for sent_id, sent_predicted_relations in zip(sent_idcs, predicted_relations):
            self.update_state(sent_id, sent_predicted_relations)

    def reset_state(self):
        self.all_correct = 0
        self.all_predicted = 0
        self.all_gold = 0

    def update_state(self, sent_id, sent_predicted_relations):
        if sent_predicted_relations is not None:
            rel_pairs = self.dependency.labeled_relations[sent_id]
            sent_predicted_relations = \
                [(d, p, self.dependency.reverse_label_map[l]) for d, p, l in sent_predicted_relations]
            self.all_gold += len(rel_pairs)
            self.all_predicted += len(sent_predicted_relations)
            self.all_correct += len(set(rel_pairs).intersection(set(sent_predicted_relations)))

    def result(self):
        if not self.all_correct:
            return 0.
        return 2. / (self.all_predicted / self.all_correct + self.all_gold / self.all_correct)


