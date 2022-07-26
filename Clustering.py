import numpy.linalg as LA
from scipy.optimize import minimize
from math import inf
from sklearn.preprocessing import normalize
from pyclustering.cluster.kmedians import kmedians
from itertools import permutations

from tqdm import tqdm
from Synthetic import *


def count_transition(x, a, y, trajectories):
    cnt = 0
    H = len(trajectories[0])
    for trajectory in trajectories:
        cnt += sum(1 for i in range(H - 2) if i % 2 == 0 and trajectory[i:i + 3] == [x, a, y])
    return cnt

def count_visitation(x, a, trajectories):
    cnt = 0
    H = len(trajectories[0])
    for trajectory in trajectories:
        cnt += sum(1 for i in range(H - 1) if i % 2 == 0 and trajectory[i:i + 2] == [x, a])
    return cnt


def count_transition_latent(s, a, v, trajectories, f):
    cnt = 0
    H = len(trajectories[0])
    for trajectory in trajectories:
        cnt += sum(1 for i in range(H - 2) if i % 2 == 0 and f[trajectory[i]] == s and trajectory[i + 1] == a and f[
            trajectory[i + 2]] == v)
    return cnt


def count_transition_mixed1(x, a, s, trajectories, f):
    cnt = 0
    H = len(trajectories[0])
    for trajectory in trajectories:
        cnt += sum(1 for i in range(H - 2) if i % 2 == 0 and trajectory[i] == x and trajectory[i + 1] == a and f[
            trajectory[i + 2]] == s)
    return cnt


def count_transition_mixed2(s, a, x, trajectories, f):
    cnt = 0
    H = len(trajectories[0])
    for trajectory in trajectories:
        cnt += sum(1 for i in range(H - 2) if
                   i % 2 == 0 and f[trajectory[i]] == s and trajectory[i + 1] == a and trajectory[i + 2] == x)
    return cnt


def low_rank(N, r=1):
    U, S, V = LA.svd(N, full_matrices=False)
    Nr = np.zeros((len(U), len(V)))
    for i in range(r):
        Nr += S[i] * np.outer(U.T[i], V[i])
    return Nr


def count_error(f, f_1, perm, n):
    cnt = 0
    for x in range(n):
        if f[x] != perm[f_1[x]]:
            cnt += 1
    return cnt


def error_rate(f, f_1, n, S):
    error = n
    for perm_ in permutations(range(S)):
        perm = {}
        for s in range(S):
            perm[s] = perm_[s]
        error = min(error, count_error(f, f_1, perm, n))

    return error / n



def init_spectral(env, trajectories):
    n, S, A, H = env.n, env.S, env.A, env.H
    T = len(trajectories)

    # Collect trimmed, low-rank approx, empirical transition matrices
    transition_matrices_before = []
    transition_matrices = []
    for a in range(A):
        # Collect empirical transition matrices
        transition_matrix_a = np.zeros([n, n])
        visitations_a = np.zeros([n])
        for x in range(n):
        # for x in tqdm(range(n)):
            visitations_a[x] = count_visitation(x, a, trajectories)
            for y in range(n):
                transition_matrix_a[x, y] = count_transition(x, a, y, trajectories)
        # Trimming!
        ratio = (T * H) / (n * A)
        num_trimmed = int(np.floor(n * np.exp(- ratio * np.log(ratio))))
        if num_trimmed > 0:
            contexts_ordered = np.argsort(visitations_a)
            contexts_trimmed = contexts_ordered[-num_trimmed:]
            for x, y in zip(contexts_trimmed, contexts_trimmed):
                transition_matrix_a[x][y] = 0

        transition_matrices_before.append(transition_matrix_a)
        # Low-rank approximation
        transition_matrices.append(low_rank(transition_matrix_a, r=S))

    M_in = np.concatenate(tuple(transition_matrices), axis=1)
    M_out = np.concatenate(tuple(transition_matrices), axis=0).T
    M = np.concatenate((M_in, M_out), axis=1)

    # l1-normalize rows
    row_sums = M.sum(axis=1)
    row_sums[row_sums == 0] = 1
    M = M / row_sums[:, np.newaxis]
    # M = normalize(M, norm='l1', axis=1)

    # S-median clustering to the rows
    initial_medians = M[:S, :]
    # initial_medians = np.random.randn(S, 2*n*A)
    kmedians_instance = kmedians(M, initial_medians)
    kmedians_instance.process()
    clusters = kmedians_instance.get_clusters()

    f_1 = {}
    for x in range(n):
        for s in range(S):
            if x in clusters[s]:
                f_1[x] = s

    return f_1


def likelihood_improvement(env, trajectories, f_1, f, num_iter=None):
    # likelihood_improvement
    n, S, A, H = env.n, env.S, env.A, env.H

    f_final = f_1
    if num_iter is None:
        num_iter = int(np.floor(np.log(n * A)))
    errors = []
    fs = []

    for _ in range(num_iter):
    # for _ in tqdm(range(num_iter)):
        # estimated latent transition matrices
        Ns = [np.zeros((S, S)) for _ in range(A)]
        for a in range(A):
            for s in range(S):
                for k in range(S):
                    Ns[a][s][k] = count_transition_latent(s, a, k, trajectories, f_final)

        # likelihood improvement
        f_ = {}
        for x in range(n):
            likelihoods = []
            for j in range(S):
                # N2 = number of visitations to j
                N2 = 0
                for a in range(A):
                    tmp = np.sum(Ns[a], axis=0)
                    N2 += tmp[j]
                likelihood = 0
                for a in range(A):
                    # N1 = number of visitations from (j, a)
                    tmp = np.sum(Ns[a], axis=1)
                    N1 = tmp[j]
                    # degenerate case
                    if N1 == 0 or N2 == 0:
                        likelihood = -inf
                    else:
                        for s in range(S):
                            # estimate of p and p_bwd
                            p_estimated = Ns[a][j][s] / N1  # ((j, a) -> s) / ((j, a) -> X)
                            p_bwd_estimated = Ns[a][s][j] / N2  # (j <- (s, a)) / (j <- X)
                            if p_estimated == 0 or p_bwd_estimated == 0:
                                likelihood = -inf
                                continue
                            # number of visitations (x, a) -> s
                            N3 = count_transition_mixed1(x, a, s, trajectories, f_final)
                            # number of visitations (s, a) -> x
                            N4 = count_transition_mixed2(s, a, x, trajectories, f_final)

                            # compute likelihood
                            likelihood += (N3 * np.log(p_estimated)) + (N4 * np.log(p_bwd_estimated))
                likelihoods.append(likelihood)
            # new cluster
            # print(likelihoods)
            f_[x] = np.argmax(likelihoods)
            fs.append(f_)
        f_final = f_
        errors.append(error_rate(f, f_final, env.n, env.S))

    return f_final, errors


if __name__ == '__main__':
    n = 100
    A = 2
    H = 100
    T = int(np.ceil(n * A * (np.log(n * A) ** 1.1)) / H)

    env_config = {'n': n, 'H': H, 'S:': 2, 'A': A,
                  'ps': [np.array([[3 / 4, 1 / 4], [1 / 4, 3 / 4]]), np.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])],
                  'qs': 'uniform'}
    env = Synthetic(env_config)
    # true clusters
    f = {}
    for s in range(env.S):
        cluster = env.partitions[s]
        for x in range(cluster.start, cluster.start + cluster.n):
            f[x] = s
    # obtain trajectories
    trajectories = generate_trajectories(T, env)

    # initial spectral clustering
    f_1 = init_spectral(env, trajectories)
    init_err_rate = error_rate(f, f_1, env.n, env.S)
    print("Error rate after initial clustering is ", init_err_rate)

    # likelihood_improvement
    # f_final, errors = likelihood_improvement(env, trajectories, f_1, f, num_iter=10)
    f_final, errors = likelihood_improvement(env, trajectories, f_1, f, num_iter=None)
    print("Final error rate is ", errors[-1])
    print("Errors along the improvement steps: ", errors)
