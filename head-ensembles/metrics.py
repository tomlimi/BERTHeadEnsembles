import numpy as np
from abc import abstractmethod
from collections import defaultdict

class Metric:

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass


class DepAcc(Metric):

    def __init__(self, dependency_relations, relation_label, dependent2parent=True):
        self.dependency_relations = [sent_relations[relation_label] for sent_relations in dependency_relations]

        self.relation_label = relation_label
        self.dependent2parent = dependent2parent

    def calculate(self, sent_idcs, matrices):
        retrieved = 0
        total = 0

        for index, matrix in zip(sent_idcs, matrices):
            if matrix is not None:
                np.fill_diagonal(matrix, 0.)
                max_row = matrix.argmax(axis=1)

                rel_pairs = self.dependency_relations[index]
                if not self.dependent2parent:
                    rel_pairs = list(map(tuple, map(reversed, rel_pairs)))
                retrieved += sum([max_row[attending] == attended for attending, attended in rel_pairs ])
                total += len(rel_pairs)

        if total == 0:
            return 0

        return retrieved / total
    
    
class UAS(Metric):
	def __init__(self, unlabeld_dependency_relations, roots):
		self.unlabeled_dependency_relations = unlabeld_dependency_relations
	
	def calculate(self, sent_idcs, matrices):
		results =[]
		for index, matrix in zip(sent_idcs, matrices):
			if matrix is not None:
				np.fill_diagonal(matrix, 0.)
				max_row = matrix.argmax(axis=1)
				
				rel_pairs = self.unlabeled_dependency_relations[index]

				correct = sum([max_row[attending] == attended for attending, attended in rel_pairs])
				results.append(correct+1/len(rel_pairs)+1)  # +1 in denominator and numerator because we use gold root
		
		return np.array(results).mean()


class LAS(Metric):
	pass


