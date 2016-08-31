import numpy as np
import re
import time
import sys
import heapq
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import threading
from io import StringIO
from multiprocessing import Pool
from multiprocessing import Lock
from multiprocessing import Manager
import queue
import os


class ConllLine:
    def root_init(self):
        self.id = 0
        self.form = '*root*'
        self.lemma = '_'
        self.cpostag = '_'
        self.postag = '_'
        self.feats = '_'
        self.head = -1
        self.deptype = 'rroot'
        self.phead = -1
        self.pdeptype = '_'

    def __str__(self):
        return '\t'.join(
            [str(self.id), self.form, self.lemma, self.cpostag, self.postag, self.feats, str(self.head), self.deptype,
             str(self.phead), self.pdeptype])

    def __init__(self, tokens=None):
        if tokens is None:
            self.root_init()
        else:
            self.id = int(tokens[0])
            self.form = tokens[1]
            self.lemma = tokens[2]
            self.cpostag = tokens[3]
            self.postag = tokens[4]
            self.feats = tokens[5]
            self.head = int(tokens[6])
            self.deptype = tokens[7]
            if len(tokens) > 8:
                self.phead = -1 if tokens[8] == '_' else int(tokens[8])
                self.pdeptype = tokens[9]
            else:
                self.phead = -1
                self.pdeptype = '_'

    tree_line_extractor = re.compile('([a-z]+)\(.+-(\d+), (.+)-(\d+)\)')

    # stanford parser tree output:  num(Years-3, Five-1)
    def from_tree_line(self, tree_line):
        self.root_init()
        tok = self.tree_line_extractor.match(tree_line)
        self.id = int(tok.group(4))
        self.form = tok.group(3)
        self.head = int(tok.group(2))
        self.deptype = tok.group(1)


# noinspection PyUnboundLocalVariable
def read_conll(conll_file, lower):
    root = ConllLine()
    words = [root]
    for line in conll_file:
        line = line.strip()
        if len(line) > 0:
            if lower:
                line = line.lower()
            tokens = line.split('\t')
            words.append(ConllLine(tokens))
        else:
            if len(words) > 1:
                yield words
                words = [root]
    if len(tokens) > 1:
        yield tokens


def normalize(m):
    norm = np.sqrt(np.sum(m * m, axis=1))
    norm[norm == 0] = 1
    return m / norm[:, np.newaxis]


def readVocab(path):
    vocab = []
    with open(path) as f:
        for line in f:
            vocab.extend(line.strip().split())
    return dict([(w, i) for i, w in enumerate(vocab)]), vocab


# noinspection PyTypeChecker,PyTypeChecker
class Embedding:
    def __init__(self, path):
        self.m = normalize(np.load(path + '.npy'))
        self.dim = self.m.shape[1]
        self.wi, self.iw = readVocab(path + '.vocab')

    def zeros(self):
        return np.zeros(self.dim)

    def dimension(self):
        return self.dim

    def __contains__(self, w):
        return w in self.wi

    def represent(self, w):
        return self.m[self.wi[w], :]

    def scores(self, vec):
        return np.dot(self.m, vec)

    def pos_scores(self, vec):
        return (np.dot(self.m, vec) + 1) / 2

    def pos_scores2(self, vec):
        scores = np.dot(self.m, vec)
        scores[scores < 0.0] = 0.0
        return scores

    def top_scores(self, scores, n=10):
        if n <= 0:
            n = len(scores)
        return heapq.nlargest(n, zip(self.iw, scores), key=lambda x: x[1])

    def closest(self, w, n=10):
        scores = np.dot(self.m, self.represent(w))
        return self.top_scores(scores, n)

    def closest_with_time(self, w, n=10):
        start = time.time()
        scores = np.dot(self.m, self.represent(w))
        end = time.time()
        #        print "\nDeltatime: %f msec\n" % ((end-start)*1000)
        return self.top_scores(scores, n), end - start

    def closest_vec(self, wordvec, n=10):
        # scores = self.m.dot(self.represent(w))
        scores = np.dot(self.m, wordvec)
        return self.top_scores(scores, n)

    #        if n <= 0:
    #            n = len(scores)
    #        return heapq.nlargest(n, zip(self.iw, scores))

    # noinspection PyTypeChecker
    def closest_vec_filtered(self, wordvec, vocab, n=10):
        scores = np.dot(self.m, wordvec)
        if n <= 0:
            n = len(scores)
        scores_words = zip(self.iw, scores)
        for i in range(0, len(scores_words)):
            if not scores_words[i][1] in vocab:
                scores_words[i] = (-1, scores_words[i][0])
        return heapq.nlargest(n, zip(self.iw, scores), key=lambda x: x[1])

    def closest_prefix(self, w, prefix, n=10):
        scores = np.dot(self.m, self.represent(w))
        scores_words = zip(self.iw, scores)
        for i in range(0, len(scores_words)):
            if not scores_words[i][1].startswith(prefix):
                scores_words[i] = (-1, scores_words[i][0])
        return heapq.nlargest(n, scores_words, key=lambda x: x[1])

    def closest_filtered(self, w, vocab, n=10):
        scores = np.dot(self.m, self.represent(w))
        scores_words = zip(self.iw, scores)
        for i in range(0, len(scores_words)):
            if not scores_words[i][1] in vocab:
                scores_words[i] = (-1, scores_words[i][0])
        return heapq.nlargest(n, scores_words, key=lambda x: x[1])

    def similarity(self, w1, w2):
        return self.represent(w1).dot(self.represent(w2))


def add_inference_result(token, weight, filtered_results, candidates_found):
    candidates_found.add(token)
    best_last_weight = filtered_results[token] if token in filtered_results else None
    if best_last_weight is None or weight > best_last_weight:
        filtered_results[token] = weight


class CsInferrer(object):
    """
    classdocs
    """

    def __init__(self):
        """
        Constructor
        """
        self.time = [0.0, 0]

    def inference_time(self, seconds):
        self.time[0] += seconds
        self.time[1] += 1

        # processing time in msec

    def msec_per_word(self):
        return 1000 * self.time[0] / self.time[1] if self.time[1] > 0 else 0.0


def vec_to_str(subvec, max_n):
    def wf2ws(weight):
        return '{0:1.5f}'.format(weight)

    sub_list_sorted = heapq.nlargest(max_n, subvec, key=lambda x: x[1])
    sub_strs = [' '.join([word, wf2ws(weight)]) for word, weight in sub_list_sorted]
    return '\t'.join(sub_strs)


class CsEmbeddingInferrer(CsInferrer):
    def __init__(self, context_math, word_path, context_path, conll_filename, top_inferences_to_analyze, WIN_SIZE):

        CsInferrer.__init__(self)
        self.context_math = context_math
        self.word_vecs = Embedding(word_path)
        self.context_vecs = Embedding(context_path)
        self.WIN_SIZE = WIN_SIZE
        self.top_inferences_to_analyze = top_inferences_to_analyze

        self.lemmas = {}
        for _w in self.word_vecs.iw:
            for wn_pos in [wordnet.NOUN, wordnet.ADJ, wordnet.VERB, wordnet.ADV]:
                self.lemmas['_'.join([_w, wn_pos])] = WordNetLemmatizer().lemmatize(_w, wn_pos)

        self.to_wordnet_pos = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

    # noinspection PyUnusedLocal,PyUnusedLocal
    def generate_inferred(self, result_vec, target_word, target_lemma, pos):
        generated_word_re = re.compile('^[a-zA-Z]+$')

        generated_results = {}
        min_weight = None
        if result_vec is not None:
            for word, weight in result_vec:
                if generated_word_re.match(word) is not None:  # make sure this is not junk
                    wn_pos = self.to_wordnet_pos[pos]
                    lemma = self.lemmas['_'.join([word, wn_pos])]
                    if word != target_word and lemma != target_lemma:
                        if word in generated_results:
                            weight = max(weight, generated_results[word])
                        generated_results[word] = weight
                        if min_weight is None:
                            min_weight = weight
                        else:
                            min_weight = min(min_weight, weight)

        if min_weight is None:
            min_weight = 0.0
        i = 0.0

        # just something to return in case not enough words were generated
        default_generated_results = ['time', 'people', 'information', 'work', 'first', 'like', 'year', 'make', 'day',
                                     'service']

        for lemma in default_generated_results:
            if len(generated_results) >= len(default_generated_results):
                break
            i -= 1.0
            generated_results[lemma] = min_weight + i

        return generated_results

    # noinspection PyShadowingNames
    def add(self, scores_vec, bal_flag):
        scores = scores_vec[0, :]
        for i in range(1, scores_vec.shape[0]):
            add_scores = (scores_vec[i, :] + 1) / 2
            if bal_flag:
                add_scores /= (scores_vec.shape[0] - 1)
            scores = np.add(scores, add_scores)

        result_vec = self.word_vecs.top_scores(scores, -1)
        return result_vec

    # noinspection PyShadowingNames
    def mult(self, scores_vec, bal_flag):

        # SUPPORT NONE TARGET

        # target_vec = self.word_vecs.represent(target)
        scores = (scores_vec[0, :] + 1) / 2
        for i in range(1, scores_vec.shape[0]):
            # if dep in self.context_vecs:
            #     dep_vec = self.context_vecs.represent(dep)
            #     mult_scores = self.word_vecs.pos_scores(dep_vec)
            #     if geo_mean_flag:
            #         mult_scores **= 1.0 / len(deps)
            #     scores = np.multiply(scores, mult_scores)
            # else:
            #     tfo.write("NOTICE: %s not in context embeddings. Ignoring.\n" % dep)
            mult_scores = (scores_vec[i, :] + 1) / 2
            if bal_flag:
                mult_scores **= 1.0 / (scores_vec.shape[0] - 1)
            scores = np.multiply(scores, mult_scores)

        result_vec = self.word_vecs.top_scores(scores, -1)
        return result_vec

    def get_deps(self, sent, target_ind):
        deps = {}

        for word_line in sent[1:]:
            parent_line = sent[word_line.head]
            # universal        if word_line.deptype == 'adpmod': # we are collapsing preps
            if word_line.deptype == 'prep':  # we are collapsing preps
                continue
            # universal       if word_line.deptype == 'adpobj' and parent_line.id != 0: # collapsed dependency
            if word_line.deptype == 'pobj' and parent_line.id != 0:  # collapsed dependency
                grandparent_line = sent[parent_line.head]
                # if grandparent_line.id != target_ind and word_line.id != target_ind:
                #     continue
                relation = "%s:%s" % (parent_line.deptype, parent_line.form)
                head = grandparent_line.form
                ih = grandparent_line.id
            else:  # direct dependency
                # if parent_line.id != target_ind and word_line.id != target_ind:
                #     continue
                head = parent_line.form
                ih = parent_line.id
                relation = word_line.deptype
            # if word_line.id == target_ind:
            #     if head not in stopwords:
            #         deps.append("I_".join((relation, head)))
            # else:
            #     if word_line.form not in stopwords:
            #         deps.append("_".join((relation, word_line.form)))
            #         #      print h,"_".join((rel,m))
            #         #      print m,"I_".join((rel,h))
            deps[(word_line.id, word_line.form)] = (ih, head, relation)

        ldeps = []
        for m in deps:
            d = []
            curr = m
            for depth in range(self.WIN_SIZE):
                d += [deps[curr], ]

                if deps[curr][:2] in deps:
                    curr = deps[curr][:2]
                else:
                    break

            flag = 0
            for w in d:
                if w[0] == target_ind:
                    flag = 1
                    break

            if m[0] != target_ind and flag != 1:
                continue

            if m[0] == target_ind:
                d1 = tuple(["I_".join((x[2], x[1])) for x in d])
                while len(d1) > 0:
                    if '::|::'.join(d1) in self.context_vecs.wi:
                        # print(m[1], '::|::'.join(d1))
                        ldeps.append('::|::'.join(d1))
                        break
                    d1 = d1[:-1]
                continue

            if d[-1][1] != '*root*':
                idx = -1
            elif len(d) > 1:
                idx = -2
                while idx >= 0 and d[idx][1] not in self.word_vecs.wi:
                    idx -= 1
            else:
                continue

            ih = d[idx][0]
            h = d[idx][1]
            d2 = [[m + (d[0][2],), ], ] + [[(d[i][0], d[i][1], d[i + 1][2]), ] for i in range(0, len(d) + idx)]
            d2 = d2[0]

            d1 = tuple(["_".join((x[2], x[1])) for x in d2])
            while len(d1) > 0:
                if ih == target_ind and '::|::'.join(d1) in self.context_vecs.wi:
                    ldeps.append('::|::'.join(d1))
                    break

                idx -= 1
                if len(d) + idx < 0:
                    break

                ih = d[idx][0]
                h = d[idx][1]
                d2 = [[m + (d[0][2],), ], ] + [[(d[i][0], d[i][1], d[i + 1][2]), ] for i in range(0, len(d) + idx)]
                d2 = d2[0]
                d1 = tuple(["_".join((x[2], x[1])) for x in d2])

        return ldeps

    def extract_contexts(self, lst_instance, conll):
        cur_sent = conll  # next(self.sents)
        cur_target_ind = lst_instance.target_ind + 1

        while cur_target_ind < len(cur_sent) and cur_sent[cur_target_ind].form != lst_instance.target:
            cur_target_ind += 1

        if cur_target_ind == len(cur_sent):
            cur_target_ind = lst_instance.target_ind
            while (cur_target_ind > 0) and (cur_sent[cur_target_ind].form != lst_instance.target):
                cur_target_ind -= 1

        if cur_target_ind == 0:
            sys.stderr.write(
                "ERROR: Couldn't find a match for target. {}: {}".format(lst_instance.target_ind, lst_instance.target))
            cur_target_ind = lst_instance.target_ind + 1

        contexts = self.get_deps(cur_sent, cur_target_ind)

        return contexts

    def find_scores(self, lst_instance, tfo, conll):
        contexts = self.extract_contexts(lst_instance, conll)

        # tfo.write("Contexts for target %s are: %s\n" % (lst_instance.target, contexts))
        contexts = [c for c in contexts]
        tfo.write("Contexts in vocabulary for target %s are: %s\n" % (lst_instance.target, contexts))

        if lst_instance.target not in self.word_vecs:
            tfo.write("ERROR: %s not in word embeddings.Trying lemma.\n" % lst_instance.target)
            if lst_instance.target_lemma not in self.word_vecs:
                tfo.write("ERROR: lemma %s also not in word embeddings. Giving up.\n" % lst_instance.target_lemma)
                return None
            else:
                target = lst_instance.target_lemma
        else:
            target = lst_instance.target

        target_vec = self.word_vecs.represent(target)

        dep_index = []
        for dep in contexts:
            dep_vec = self.context_vecs.represent(dep)
            # scores = np.vstack((scores, self.word_vecs.scores(dep_vec)))
            dep_index.append(self.context_vecs.wi[dep])
        dep_vecs = self.context_vecs.m[dep_index, :]

        scores = self.word_vecs.scores(np.vstack((target_vec, dep_vecs)).T).T

        return scores

    # noinspection PyShadowingNames
    def find_inferred(self, scores, tfo, measure, target):
        result_vec = None
        if measure == 'add':
            result_vec = self.add(scores, False)
        elif measure == 'baladd':
            result_vec = self.add(scores, True)
        elif measure == 'mult':
            result_vec = self.mult(scores, False)
        elif measure == 'balmult':
            result_vec = self.mult(scores, True)
        elif measure == 'none':
            result_vec = self.word_vecs.closest(target, -1)
        else:
            raise Exception('Unknown context math: %s' % measure)

        if result_vec is not None:
            tfo.write("Top most similar embeddings: " + vec_to_str(result_vec, self.top_inferences_to_analyze) + '\n')
        else:
            tfo.write("Top most similar embeddings: " + " contexts: None\n")

        return result_vec

    def filter_inferred(self, result_vec, candidates, pos):

        filtered_results = {}
        candidates_found = set()

        if result_vec is not None:
            for word, weight in result_vec:
                wn_pos = self.to_wordnet_pos[pos]

                if '_'.join([word, wn_pos]) not in self.lemmas:
                    print('_'.join([word, wn_pos]), 'not in lemmas')
                else:
                    lemma = self.lemmas['_'.join([word, wn_pos])]
                    if lemma in candidates:
                        add_inference_result(lemma, weight, filtered_results, candidates_found)
                    if lemma.title() in candidates:
                        add_inference_result(lemma.title(), weight, filtered_results, candidates_found)
                if word in candidates:  # there are some few cases where the candidates are not lemmatized
                    add_inference_result(word, weight, filtered_results, candidates_found)
                if word.title() in candidates:  # there are some few cases where the candidates are not lemmatized
                    add_inference_result(word.title(), weight, filtered_results, candidates_found)

                    # assign negative weights for candidates with no score
                    # they will appear last sorted according to their unigram count
                    #        candidates_left = candidates - candidates_found
                    #        for candidate in candidates_left:
                    #            count = self.w2counts[candidate] if candidate in self.w2counts else 1
                    #            score = -1 - (1.0/count) # between (-1,-2]
                    #            filtered_results[candidate] = score

        return filtered_results


# noinspection PyShadowingNames
def read_candidates(candidates_file):
    target2candidates = {}
    # finally.r::eventually;ultimately
    with open(candidates_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            segments = line.split('::')
            target = segments[0]
            candidates = set(segments[1].strip().split(';'))
            target2candidates[target] = candidates
    return target2candidates


CONTEXT_TEXT_BEGIN_INDEX = 3
TARGET_INDEX = 2
from_lst_pos = {'j': 'J', 'a': 'J', 'v': 'V', 'n': 'N', 'r': 'R'}


class ContextInstance(object):
    def __init__(self, line, no_pos_flag):
        """
        Constructor
        """
        self.line = line
        tokens1 = line.split("\t")
        self.target_ind = int(tokens1[TARGET_INDEX])
        self.words = tokens1[3].split()
        self.target = self.words[self.target_ind]
        self.full_target_key = tokens1[0]
        self.pos = self.full_target_key.split('.')[-1]
        self.target_key = '.'.join(self.full_target_key.split('.')[:2])  # remove suffix in cases of bar.n.v
        self.target_lemma = self.full_target_key.split('.')[0]
        self.target_id = tokens1[1]
        if self.pos in from_lst_pos:
            self.pos = from_lst_pos[self.pos]
        self.target_pos = '.'.join([self.target, '*']) if no_pos_flag is True else '.'.join([self.target, self.pos])

    def get_neighbors(self, window_size):
        tokens = self.line.split()[3:]

        if window_size > 0:
            start_pos = max(self.target_ind - window_size, 0)
            end_pos = min(self.target_ind + window_size + 1, len(tokens))
        else:
            start_pos = 0
            end_pos = len(tokens)

        neighbors = tokens[start_pos:self.target_ind] + tokens[self.target_ind + 1:end_pos]
        return neighbors

    def decorate_context(self):
        tokens = self.line.split('\t')
        words = tokens[CONTEXT_TEXT_BEGIN_INDEX].split()
        words[self.target_ind] = '__' + words[self.target_ind] + '__'
        tokens[CONTEXT_TEXT_BEGIN_INDEX] = ' '.join(words)
        return '\t'.join(tokens) + "\n"


def vec_to_str_generated(subvec, max_n):
    sub_list_sorted = heapq.nlargest(max_n, subvec, key=lambda x: x[1])
    sub_strs = [word for word, weight in sub_list_sorted]
    return ';'.join(sub_strs)


# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames
def thread_run(file_objects, inferrer, target2candidates, context_lines, measures):
    for context_line, conll in context_lines:
        # print(context_line)
        lst_instance = ContextInstance(context_line, False)

        tfo_s = StringIO()

        scores = inferrer.find_scores(lst_instance, tfo_s, conll)

        for measure in measures:
            result_vec = inferrer.find_inferred(scores, tfo_s, measure, lst_instance.target)

            generated_results = inferrer.generate_inferred(result_vec, lst_instance.target, lst_instance.target_lemma,
                                                           lst_instance.pos)

            filtered_results = inferrer.filter_inferred(result_vec, target2candidates[lst_instance.target_key],
                                                        lst_instance.pos)

            file_objects[measure]['results'].write("\nTest context:\t")
            file_objects[measure]['results'].write(lst_instance.decorate_context())
            file_objects[measure]['results'].write(tfo_s.getvalue())
            file_objects[measure]['results'].write("GENERATED\t"
                                                   + ' '.join([lst_instance.full_target_key,
                                                               lst_instance.target_id])
                                                   + " ::: " + vec_to_str_generated(generated_results.items(), 10)
                                                   + '\n')

            file_objects[measure]['oot'].write(' '.join([lst_instance.full_target_key, lst_instance.target_id])
                                               + " ::: " + vec_to_str_generated(generated_results.items(), 10) + "\n")

            file_objects[measure]['best'].write(' '.join([lst_instance.full_target_key, lst_instance.target_id])
                                                + " :: " + vec_to_str_generated(generated_results.items(), 1) + "\n")

            file_objects[measure]['results'].write("RANKED\t"
                                                   + ' '.join([lst_instance.full_target_key, lst_instance.target_id])
                                                   + "\t" + vec_to_str(filtered_results.items(), len(filtered_results))
                                                   + "\n")

            file_objects[measure]['ranked'].write("RANKED\t"
                                                  + ' '.join([lst_instance.full_target_key, lst_instance.target_id])
                                                  + "\t" + vec_to_str(filtered_results.items(), len(filtered_results))
                                                  + "\n")


if __name__ == '__main__':

    oper = sys.argv[1]
    ipath = sys.argv[2]
    epath = sys.argv[3]
    opath = sys.argv[4]

    try:
        dataset = sys.argv[5]
    except IndexError:
        dataset = 'lst'

    # noinspection PyBroadException
    try:
        WIN_SIZE = int(sys.argv[6])
    except:
        WIN_SIZE = 1

    class Arg:
        pass

    args = Arg()
    args.vocabfile = None
    args.contextmath = oper

    args.embeddingpath = ipath + '/vecs'
    args.embeddingpathc = ipath + '/contexts'

    if dataset == 'lst':
        args.testfileconll = epath + '/lst.conll'
        args.candidatesfile = epath + '/lst.candidates'
        args.testfile = epath + '/lst'
        num_eval_lines = 2010
    elif dataset == 'cic':
        args.testfileconll = epath + '/cic.conll'
        args.candidatesfile = epath + '/cic.candidates'
        args.testfile = epath + '/coinco_all.no_problematic.preprocessed'
        num_eval_lines = 15415
    else:
        raise AssertionError('dataset not supported')

    args.resultsfile = opath + '/results'
    args.topgenerated = 10

    inferrer = CsEmbeddingInferrer(args.contextmath, args.embeddingpath,
                                   args.embeddingpathc, args.testfileconll, 10, WIN_SIZE)

    target2candidates = read_candidates(args.candidatesfile)

    sents = read_conll(open(args.testfileconll), True)

    file_objects = {'test': open(args.testfile, 'r', encoding='iso-8859-1')}

    measures = ['add', 'baladd', 'mult', 'balmult']

    for measure in measures:
        file_objects[measure] = {}

        out_dir = os.path.join(opath, measure)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        file_objects[measure]['results'] = open(out_dir + '/results', 'w', encoding='iso-8859-1')
        file_objects[measure]['ranked'] = open(out_dir + '/results.ranked', 'w', encoding='iso-8859-1')
        file_objects[measure]['oot'] = open(out_dir + '/results.generated.oot', 'w', encoding='iso-8859-1')
        file_objects[measure]['best'] = open(out_dir + '/results.generated.best', 'w', encoding='iso-8859-1')

    num_lines = 0
    start_time = time.time()

    context_lines = ()

    lines = 0
    while True:
        context_line = file_objects['test'].readline()

        if not context_line:
            break

        lines += 1
        conll = next(sents)

        context_lines += ((context_line, conll),)
        thread_run(file_objects, inferrer, target2candidates, context_lines, measures)
        context_lines = ()

        num_lines += 1
        if num_lines % 100 == 0:
            end_time = time.time()
            t = (100 / (end_time - start_time) if (end_time - start_time) > 0 else 0)
            print("Read {} lines,\tSpeed={:.2f} lines/sec".format(num_lines, t))
            sys.stdout.flush()
            start_time = time.time()

            for measure in measures:
                file_objects[measure]['results'].flush()
                file_objects[measure]['ranked'].flush()
                file_objects[measure]['oot'].flush()
                file_objects[measure]['best'].flush()

    file_objects['test'].close()

    for measure in measures:
        file_objects[measure]['results'].close()
        file_objects[measure]['ranked'].close()
        file_objects[measure]['oot'].close()
        file_objects[measure]['best'].close()
