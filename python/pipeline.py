### USAGE EXAMPLES
#
# To run pipeline:
#
# Example:
#
# python3 pipeline.py ./data/20news/ mean-sinkhorn-exact 1000-40-1 9
#
# This example runs mean with 1000 output candidates, then sinkhorn with 9 iterations and 40 output candidates, then exact computation with 1 output candidates.
#
# Arguments:
#   First - path to data folder (see below for required data folder content and format)
#   Second - list (separated by hyphens) of methods in the pipeline
#   Third - list (separated by hyphers) of  numbers of candidates per method
#   Fourth (optional, default=1) - integer parameter that will be used as number of iterations for sinkhorn and lcwmd
#
# Details:
#
# Data folder:
# Needs to contain 4 files: vocab.npy, dataset.npy, queries.npy, answers.npy.
# vocab.npy: V x D array where V is the number of points in the ground metric (i.e., words in the dictionary), and D is their dimension.
#            Thus row i is the d-dimensional embedding of word i.
# dataset.npy: 1-dimensional array of length N of lists, where each list is a document, represented as a list of word indices.
#              Each document [i_1, i2, ...] represents a uniform distribution over the words vocab[i1], vocab[i2], ...
# queries.npy: 1-dimensional array of length Q of lists.
#              Same as dataset.npy, only those documents would be used as queries (for whom we search for nearest neighbors in the dataset).
# answers.npy: N x Q. Note that the number of rows equals the length of dataset.npy, and the number of queries equals the length of queries.npy.
#              Each column corresponds to a query, and the content of the column is the sorted list of its nearest neighbors in the dataset.
#              For example answers[0,q] is the index (in dataset.npy) of the nearest neighbor of query q, that is, dataset[answers[0,q]] is the
#              nearest neighbor of queries[q]. Similarly answers[1,q] is the index of its second-nearest-neighbor, and so on.
#
# List of methods and candidate numbers:
# The supported methods: mean, overlap, quadtree, flowtree, rwmd, lcwmd, sinkhorn, exact. 
# "rwmd" is Relaxed-WMD (Kusner et al. 2015). "lcwmd" is ACT (Atasu and Mittelholzer 2019). "sinkhorn" is Sinkhorn (Cuturi 2013). "exact" uses the POT library.
#
# Parameter for sinkhorn/lcwmd: The parameter will be used in both if both appear in the pipeline; currently no way to set different parameters for them.
#
###


import sys
import time
import numpy as np
import algorithms as test

### Pipeline parameters

data_folder = sys.argv[1]

test.load_data(data_folder)

methods = sys.argv[2].split("-")
ncs = [int(x) for x in sys.argv[3].split("-")]

# Parameter for Sinkhorn or LC-WMD
method_param = 1
if len(sys.argv) > 4:
    method_param = int(sys.argv[4])

target_accuracy = 0.9
sort_flag = True

### Query set

query_idx = np.arange(1000)
qn = len(query_idx)
queries = test.queries[query_idx]
queries_modified = [test.queries_modified[i] for i in query_idx]
answers = test.answers[:,query_idx]


### Unified interfaces for all methods

def call_method_weighted(fnc, q_index, input_idx, nc):
    result = np.zeros(nc, dtype=np.int32)
    score_result = np.zeros(nc, dtype=np.float32)
    fnc(queries_modified[q_index], input_idx, result, score_result, sort_flag)
    return result

def call_method_uniform(fnc, q_index, input_idx, nc, method_param=-1):
    result = np.zeros(nc, dtype=np.int32)
    score_result = np.zeros(nc, dtype=np.float32)
    if method_param == -1:
        fnc(queries[q_index], input_idx, result, score_result, sort_flag)
    else:
        fnc(queries[q_index], input_idx, result, score_result, sort_flag, method_param)
    return result

def call_exact(q_index, input_idx):
    return [test.exact_emd(queries[q_index], input_idx)]

fdic = {}
fdic["mean"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.means_rank, q_index, input_idx, nc)
fdic["overlap"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.overlap_rank, q_index, input_idx, nc)
fdic["quadtree"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.quadtree_rank, q_index,input_idx,  nc)
fdic["flowtree"] = lambda q_index, input_idx, nc:call_method_weighted(test.solver.flowtree_rank, q_index, input_idx, nc)
fdic["rwmd"] = lambda q_index, input_idx, nc:call_method_uniform(test.rwmd, q_index, input_idx, nc)
fdic["lcwmd"] = lambda q_index, input_idx, nc:call_method_uniform(test.lc_wmd, q_index, input_idx, nc, method_param)
fdic["sinkhorn"] = lambda q_index, input_idx, nc:call_method_uniform(test.sinkhorn, q_index, input_idx, nc, method_param)
fdic["exact"] = lambda q_index, input_idx, nc:call_exact(q_index, input_idx)

### Main

accuracy = 0
accs = np.zeros(len(methods))
total_time = 0
clength = 0
input_idx = None
orig_input_idx = np.zeros(len(test.dataset), dtype=np.int32)
for i in range(len(test.dataset)):
    orig_input_idx[i] = i

start = time.time()
for q in range(qn):
    input_idx = orig_input_idx
    for m in range(len(methods)):
        input_idx = fdic[methods[m]](q, input_idx, ncs[m])
        if m==0:
            clength += np.mean([len(test.dataset[j]) for j in input_idx])
        if answers[0,q] in input_idx:
            accs[m] += 1
    if answers[0,q] in input_idx:
        accuracy += 1
    total_time = time.time() - start
    # At the end of each iteration of the main loop, input_idx is a list of indices (in the dataset) of the final candidates
    # for the current query, of the length specified by the final number in the input list of candidate numbers (third parameter).
    # As an example let us print the top-10 nearest neighbors found by the pipeline for each of the first three queries:
    if q < 3:
        print("Nearest neighbors found by pipeline for query " + str(q) + ": " + str(input_idx))

print("=== Output ===")
print("Total time: ", total_time*1./qn)
print("Final accuracy:", accuracy*1./qn)
print("Accuracy after each stage of the pipeline:", [acc*1./qn for acc in accs])

# Output in one conveniently formatted "|"-delimited line:
# print(" | ".join([str(x) for x in [methods, ncs, method_param, total_time*1./qn, accuracy*1./qn, clength*1./qn] + [acc*1./qn for acc in accs]]))




