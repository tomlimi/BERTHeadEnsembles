from collections import defaultdict

from unidecode import unidecode


class Dependency():

    pos_labels = ('ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                  'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X')

    label_map = {'acl': 'adj-clause',
                 'advcl': 'adv-clause',
                 'advmod': 'adv-modifier',
                 'amod': 'adj-modifier',
                 'appos': 'apposition',
                 'aux': 'auxiliary',
                 'xcomp': 'clausal',
                 'parataxis': 'parataxis',
                 'ccomp': 'clausal',
                 'compound': 'compound',
                 'conj': 'conjunct',
                 'cc': 'cc',
                 'csubj': 'clausal-subject',
                 'det': 'determiner',
                 'nmod': 'noun-modifier',
                 'nsubj': 'subject',
                 'nummod': 'num-modifier',
                 'obj': 'object',
                 'iobj': 'object',
                 'punct': 'punctuation',
                 'case': 'case',
                 'mark': 'mark'}

    LABEL_OTHER = 'other'
    LABEL_ALL = 'all'

    CONLLU_ID = 0
    CONLLU_ORTH = 1
    CONLLU_POS = 3
    CONLLU_HEAD = 6
    CONLLU_LABEL = 7

    def __init__(self, conll_file, bert_wordpiece_file):

        self.tokens = []
        self.relations = []
        self.wordpieces2tokens = []

        self.read_conllu(conll_file)
        self.group_wordpieces(bert_wordpiece_file)

    @classmethod
    def transform_label(cls, label):
        label = label.split(':')[0]  # to cope with nsubj:pass for instance
        if label in cls.label_map:
            label = cls.label_map[label]
        else:
            label = cls.LABEL_OTHER
        return label

    def read_conllu(self, conll_file_path):
        sentence_relations = defaultdict(list)
        sentence_tokens = []

        with open(conll_file_path, 'r') as in_conllu:
            sentid = 0
            for line in in_conllu:
                if line == '\n':
                    self.relations.append(sentence_relations)
                    sentence_relations = defaultdict(list)
                    self.tokens.append(sentence_tokens)
                    sentence_tokens = []
                    sentid += 1
                elif line.startswith('#'):
                    continue
                else:
                    fields = line.strip().split('\t')
                    if fields[self.CONLLU_ID].isdigit():
                        head_id = int(fields[self.CONLLU_HEAD]) -1
                        dep_id = int(fields[self.CONLLU_ID]) -1
                        label = self.transform_label(fields[self.CONLLU_LABEL])
                        pos_tag = fields[self.CONLLU_POS]
                        if head_id != 0:
                            sentence_relations[label].append((dep_id, head_id))
                            sentence_relations[self.LABEL_ALL].append((dep_id, head_id))

                        sentence_tokens.append(fields[self.CONLLU_ORTH])

    def group_wordpieces(self, wordpieces_file):
        '''
        Joins wordpices of tokens, so that they correspond to the tokens in conllu file.

        :param wordpieces_all: lists of BPE pieces for each sentence
        :return: group_ids_all list of grouped token ids, e.g. for a BPE sentence:
        "Mr. Kowal@@ ski called" joined to "Mr. Kowalski called" it would be [[0], [1, 2], [3]]
        '''

        with open(wordpieces_file, 'r') as in_file:
            wordpieces = [wp_sentence.strip().split() for wp_sentence in in_file.readlines()]

        grouped_ids_all = []
        tokens_out_all = []
        idx = 0
        for wordpieces, conllu_tokens in zip(wordpieces, self.tokens):
            conllu_id = 0
            curr_token = ''
            grouped_ids = []
            tokens_out = []
            wp_ids = []
            for wp_id, wp in enumerate(wordpieces):
                wp_ids.append(wp_id)
                if wp.endswith('@@'):
                    curr_token += wp[:-2]
                else:
                    curr_token += wp
                if unidecode(curr_token).lower() == unidecode(conllu_tokens[conllu_id]).lower():
                    grouped_ids.append(wp_ids)
                    wp_ids = []
                    tokens_out.append(curr_token)
                    curr_token = ''
                    conllu_id += 1
            try:
                assert conllu_id == len(conllu_tokens), f'{idx} \n' \
                                                        f'bert count {conllu_id} tokens{tokens_out} \n' \
                                                        f'conllu count {len(conllu_tokens)}, tokens {conllu_tokens}'
            except AssertionError:
                self.wordpieces2tokens.append(None)
            else:
                self.wordpieces2tokens.append(grouped_ids)
            idx += 1


# def define_labels(consider_directionality):
# 	labels_raw = list(set(label_map.values())) + ['all', 'other']
# 	global labels
# 	if consider_directionality:
# 		labels = [ar + '-d2p' for ar in labels_raw]
# 		labels.extend([ar + '-p2d' for ar in labels_raw])
# 	else:
# 		labels = labels_raw

# def conllu2freq_frame(conllu_file):
# 	dependency_pos_freq = defaultdict(lambda: pos_dict(pos_labels))
# 	relation_labeled = read_conllu_labeled(conllu_file)
# 	for sent_rels in relation_labeled:
# 		for dep, head, label, pos in sent_rels:
# 			if label != 'root':
# 				label = transform_label(label)
# 				dependency_pos_freq['all-d2p'][(pos, sent_rels[head][3])] += 1
# 				dependency_pos_freq[label + '-d2p'][(pos, sent_rels[head][3])] += 1
# 				dependency_pos_freq['all-p2d'][(sent_rels[head][3], pos)] += 1
# 				dependency_pos_freq[label + '-p2d'][(sent_rels[head][3], pos)] += 1
#
# 	pos_frame = pd.DataFrame.from_dict(dependency_pos_freq)
# 	pos_frame = pos_frame / pos_frame.sum(axis=0)[None, :]
# 	pos_frame.fillna(0, inplace=True)
#
# 	return pos_frame.to_dict()
#
#
# def conllu2dict(relations_labeled, directional=False):
# 	res_relations = []
#
# 	for sentence_rel_labeled in relations_labeled:
# 		sentence_rel = defaultdict(list)
# 		for dep, head, label, _ in sentence_rel_labeled:
# 			add_dependency_relation(sentence_rel, head, dep, label, directional)
# 		res_relations.append(sentence_rel)
# 	return res_relations