'''
See following paper for quick description of GAP:
http://aclweb.org/anthology//P/P10/P10-1097.pdf
'''

from operator import itemgetter
from random import shuffle
import copy

class GeneralizedAveragePrecision(object):

    @staticmethod
    def accumulate_score(gold_vector):
        accumulated_vector = []
        accumulated_score = 0
        for (key, score) in gold_vector:
            accumulated_score += float(score)
            accumulated_vector.append([key, accumulated_score])
        return accumulated_vector



    '''
    gold_vector: a vector of pairs (key, score) representing all valid results
    evaluated_vector: a vector of pairs (key, score) representing the results retrieved by the evaluated method
    gold_vector and evaluated vector don't need to include the same keys or be in the same length
    '''

    @staticmethod
    def calc(gold_vector, evaluated_vector, random=False):
        gold_map = {}
        for [key, value] in gold_vector:
            gold_map[key]=value
        sorted_gold_vector = sorted(gold_vector, key=itemgetter(1), reverse=True)
        gold_vector_accumulated = GeneralizedAveragePrecision.accumulate_score(sorted_gold_vector)


        ''' first we use the eval score to sort the eval vector accordingly '''
        if random is False:
            sorted_evaluated_vector = sorted(evaluated_vector, key=itemgetter(1), reverse=True)
        else:
            sorted_evaluated_vector = copy.copy(evaluated_vector)
            shuffle(sorted_evaluated_vector)
        sorted_evaluated_vector_with_gold_scores = []
        ''' now we replace the eval score with the gold score '''
        for (key, score) in sorted_evaluated_vector:
            if (key in gold_map.keys()):
                gold_score = gold_map.get(key)
            else:
                gold_score = 0
            sorted_evaluated_vector_with_gold_scores.append([key, gold_score])
        evaluated_vector_accumulated = GeneralizedAveragePrecision.accumulate_score(sorted_evaluated_vector_with_gold_scores)

        ''' this is sum of precisions over all recall points '''
        i = 0
        nominator = 0.0
        for (key, accum_score) in evaluated_vector_accumulated:
            i += 1
            if (key in gold_map.keys()) and (gold_map.get(key) > 0):
                nominator += accum_score/i

        ''' this is the optimal sum of precisions possible based on the gold standard ranking '''
        i = 0
        denominator = 0
        for (key, accum_score) in gold_vector_accumulated:
            if gold_map.get(key) > 0:
                i += 1
                denominator += accum_score/i

        if (denominator == 0.0):
            gap = -1
        else:
            gap = nominator/denominator

        return gap


    @staticmethod
    def calcTopN(gold_vector, evaluated_vector, n, measure_type):
        gold_map = {}
        for [key, value] in gold_vector:
            gold_map[key]=value
        gold_vector_sorted = sorted(gold_vector, key=itemgetter(1), reverse=True)
        gold_top_score_sum = sum([float(score) for (key, score) in gold_vector_sorted[0:n]])

        evaluated_top_score_sum = 0
        sorted_evaluated_vector = sorted(evaluated_vector, key=itemgetter(1), reverse=True)
        for (key, score) in sorted_evaluated_vector[0:n]:
            if key in gold_map:
                gold_score = gold_map[key]
            else:
                gold_score = 0
            evaluated_top_score_sum += float(gold_score)

        if measure_type == 'sap' or measure_type == 'wap':
            denominator = n
        else:
            denominator = gold_top_score_sum

        return evaluated_top_score_sum/denominator


'''
Used to compute GAP score for the LST ranking task

'''

import sys
import random
import re

#take.v 25 :: consider 2;accept 1;include 1;think about 1;
def read_gold_line(gold_line, ignore_mwe):
    segments = gold_line.split("::")
    instance_id = segments[0].strip()
    gold_weights = []
    line_candidates = segments[1].strip().split(';')
    for candidate_count in line_candidates:
        if len(candidate_count) > 0:
            delimiter_ind = candidate_count.rfind(' ')
            candidate = candidate_count[:delimiter_ind]
            if ignore_mwe and ((len(candidate.split(' '))>1) or (len(candidate.split('-'))>1)):
                continue
            count = candidate_count[delimiter_ind:]
            try:
                gold_weights.append((candidate, int(count)))
            except ValueError as e:
                print(e)
                print(gold_line)
                print("cand=%s count=%s" % (candidate,count))
                sys.exit(1)

    return instance_id, gold_weights

#RESULT    find.v 71    show 0.34657
def read_eval_line(eval_line):
    eval_weights = []
    segments = eval_line.split("\t")
    instance_id = segments[1].strip()
    for candidate_weight in segments[2:]:
        if len(candidate_weight) > 0:
            delimiter_ind = candidate_weight.rfind(' ')
            candidate = candidate_weight[:delimiter_ind]
            weight = candidate_weight[delimiter_ind:]
            if ignore_mwe and ((len(candidate.split(' '))>1) or (len(candidate.split('-'))>1)):
                continue
            try:
                eval_weights.append((candidate, float(weight)))
            except:
                print("Error appending: %s %s" % (candidate, weight))

    return instance_id, eval_weights


if __name__ == '__main__':

    epath = sys.argv[1]
    opath = sys.argv[2]

    try:
        dataset = sys.argv[3]
    except IndexError:
        dataset = 'lst'

    try:
        measure = sys.argv[4]
    except IndexError:
        measure = None

    if dataset == 'lst':
        gfile = epath + '/lst.gold'
    elif dataset == 'cic':
        gfile = epath + '/cic.gold'
    else:
        raise AssertionError('dataset not supported')

    if measure is None:
        measures = ['add', 'baladd', 'mult', 'balmult']
    else:
        measures = [measure, ]

    for measure in measures:
        resultsfile = opath + '/' + measure + '/results.ranked'
        out = opath + '/' + measure + '/gap.out'

        gold_file = open(gfile, 'r', encoding='iso-8859-1')
        eval_file = open(resultsfile, 'r', encoding='iso-8859-1')
        out_file = open(out, 'w')

        ignore_mwe = True
        randomize = False

        gold_data = {}
        eval_data = {}

        i=0
        sum_gap = 0.0
        for eval_line in eval_file:
            eval_instance_id, eval_weights = read_eval_line(eval_line)
            eval_data[eval_instance_id] = eval_weights

        for gold_line in gold_file:
            gold_instance_id, gold_weights = read_gold_line(gold_line, ignore_mwe)
            gold_data[gold_instance_id] = gold_weights

        ignored = 0
        for gold_instance_id, gold_weights in gold_data.items():
            eval_weights = eval_data[gold_instance_id]
            gap = GeneralizedAveragePrecision.calc(gold_weights, eval_weights, randomize)
            if (gap < 0): # this happens when there is nothing left to rank after filtering the multi-word expressions
                ignored += 1
                continue
            out_file.write(gold_instance_id + "\t" + str(gap) + "\n")
            i += 1
            sum_gap += gap

        mean_gap = sum_gap/i
        out_file.write("\ngold_data %d eval_data %d\n" % (len(gold_data),len(eval_data)))
        out_file.write("\nRead %d test instances\n" % i)
        out_file.write("\nIgnored %d test instances (couldn't compute gap)\n" % ignored)
        out_file.write("\nMEAN_GAP\t" + str(mean_gap) + "\n")


        gold_file.close()
        eval_file.close()
        out_file.close()
