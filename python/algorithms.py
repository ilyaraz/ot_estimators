import numpy as np
import ot_estimators as ote
import sys
import time
import ot
import os

vocab = None
dataset = None
queries = None
answers = None
dataset_modified = None
queries_modified = None
solver = None
scores = None
dataset_prep = None
dataset_dn = None
dataset_cap = None
queries_prep = None
queries_qn = None
qt = None
query_cap = None

def load_data(data_folder):
    global vocab
    global dataset
    global queries
    global answers
    global dataset_modified
    global queries_modified
    global solver
    global scores
    global dataset_prep
    global dataset_dn
    global dataset_cap

    vocab = np.load(os.path.join(data_folder, 'vocab.npy'))
    vocab = vocab.astype(np.float32)
    dataset = np.load(os.path.join(data_folder, 'dataset.npy'), allow_pickle=True)
    queries = np.load(os.path.join(data_folder, 'queries.npy'), allow_pickle=True)
    answers = np.load(os.path.join(data_folder, 'answers.npy'))

    dataset_modified = [[(t, float(1) / len(row)) for t in row] for row in dataset]
    queries_modified = [[(t, float(1) / len(row)) for t in row] for row in queries]

    solver = ote.OTEstimators()
    solver.load_vocabulary(vocab)
    solver.load_dataset(dataset_modified)

    scores = np.zeros(len(dataset), dtype=np.float32)

    dataset_prep = []
    dataset_dn = []
    dataset_cap = []
    for i in range(len(dataset)):
        cur = []
        for j in dataset[i]:
            cur.append(vocab[j])
        cur = np.vstack(cur)
        dataset_prep.append(cur)
        dataset_dn.append(np.linalg.norm(cur, axis=1).reshape(-1, 1)**2)
        dataset_cap.append(np.array([1.0 / len(dataset[i]) for j in range(len(dataset[i]))], dtype=np.float64))

def load_query(query):
    global queries_prep
    global queries_qn
    global qt
    global query_cap
    queries_prep = []
    for j in query:
        queries_prep.append(vocab[j])
    queries_prep = np.vstack(queries_prep)
    queries_qn = np.linalg.norm(queries_prep, axis=1).reshape(1, -1)**2
    qt = np.transpose(queries_prep)
    query_cap = np.array([1.0 / len(query) for j in range(len(query))], dtype=np.float64)

def get_distance_matrix(point_id):
    dm = dataset_dn[point_id] + queries_qn - 2.0 * np.dot(dataset_prep[point_id], qt)
    dm[dm < 0.0] = 0.0
    dm = np.sqrt(dm)
    return dm

def exact_emd(query, ids):
    load_query(query)
    best = 1e100
    best_id = -1
    for j in ids:
        dm = get_distance_matrix(j)
        dm = dm.astype(np.float64)
        emd_score = ot.lp.emd2(dataset_cap[j], query_cap, dm)
        if emd_score < best:
            best = emd_score
            best_id = j
    return best_id

def rwmd(query, ids, id_result, score_result, to_sort):
    load_query(query)
    k1 = ids.shape[0]
    k2 = result.shape[0]
    for i in range(k1):
        dm = get_distance_matrix(ids[i])
        scores[i] = max(np.mean(dm.min(axis=0)), np.mean(dm.min(axis=1)))
    solver.select_topk(ids, scores, id_result, score_result, to_sort)

def lc_wmd_cost(dist, k):
    if dist.shape[0] > dist.shape[1]:
        dist = dist.T
    s1 = dist.shape[0]
    s2 = dist.shape[1]
    cost1 = np.mean(dist.min(axis=0))
    if s1 == s2:
        cost2 = np.mean(dist.min(axis=1))
        return max(cost1, cost2)
    k = min(k, int(np.floor(s2/s1)), s2-1)
    remainder = (1./s1) - k*(1./s2)
    pdist = np.partition(dist, k, axis=1)
    cost2 = (np.sum(pdist[:,:k]) * 1./s2) + (np.sum(pdist[:,k]) * remainder)
    return max(cost1, cost2)

def lc_wmd(query, ids, id_result, score_result, to_sort, k_param=1):
    load_query(query)
    k1 = ids.shape[0]
    k2 = result.shape[0]
    for i in range(k1):
        dm = get_distance_matrix(ids[i])
        scores[i] = lc_wmd_cost(dm, k_param)
    solver.select_topk(ids, scores, id_result, score_result, to_sort)

def sinkhorn_cost(dist, n_iter=1):
    eta = 30

    A = np.exp(-eta*dist/dist.max())

    c = np.ones((1, dist.shape[1]), dtype=np.float32) / dist.shape[1]
    left_cap = np.ones((dist.shape[0], 1), dtype=np.float32) / dist.shape[0]

    for iii in range(n_iter):
        A *= (left_cap/np.sum(A, axis=1, keepdims=True))
        A *= (c/np.sum(A, axis=0, keepdims=True))
    x = left_cap/A.sum(1, keepdims=True)
    x[x>1] = 1.
    A = x*A
    y = c/A.sum(0, keepdims=True)
    y[y>1] = 1.
    A *= y
    err_r = (left_cap-A.sum(1, keepdims=True))
    err_r_t = (left_cap-A.sum(1, keepdims=True)).transpose()
    err_c = (c-A.sum(0, keepdims=True))
    A += np.matmul(err_r, err_c) /(np.abs(err_r_t)).sum()

    cost = (A*dist).sum()
    return cost

def sinkhorn(query, ids, id_result, score_result, to_sort, n_iter=1):
    load_query(query)
    k1 = ids.shape[0]
    k2 = id_result.shape[0]
    for i in range(k1):
        dm = get_distance_matrix(ids[i])
        scores[i] = sinkhorn_cost(dm, n_iter)
    solver.select_topk(ids, scores, id_result, score_result, to_sort)
