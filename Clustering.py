from scipy.optimize import minimize
from math import inf
from pyclustering.cluster.kmedians import kmedians
from itertools import product, permutations

from tqdm import tqdm
from Synthetic import *


def empirical_transitions(env, trajectories):
    n, A = env.n, env.A
    H = len(trajectories[0])
    output = [cp.zeros((n, n)) for _ in range(A)]
    for trajectory in trajectories:
        for i in range(0, H - 2, 2):
            output[trajectory[i + 1]][trajectory[i]][trajectory[i + 2]] += 1
    return output


def cluster_list(f, s):
    xs_s = []
    for item in list(f.items()):
        if item[1] == s:
            xs_s.append(item[0])
    return xs_s


def empirical_latent_transitions(env, empirical_transitions, f):
    S, A = env.S, env.A
    output = [cp.zeros((S, S)) for _ in range(A)]

    clustered_contexts = []
    for s in range(S):
        clustered_contexts.append(cluster_list(f, s))

    for a in range(A):
        for s, v in product(range(S), repeat=2):
            xs_s, xs_v = clustered_contexts[s], clustered_contexts[v]
            output[a][s][v] = cp.sum(empirical_transitions[a][cp.ix_(xs_s, xs_v)])
    return output


def count_transition_mixed1(x, s, empirical_transition_a, f):
    return cp.sum(empirical_transition_a[cp.ix_([x], cluster_list(f, s))])


def count_transition_mixed2(s, x, empirical_transition_a, f):
    return cp.sum(empirical_transition_a[cp.ix_(cluster_list(f, s), [x])])


def low_rank(N, r=1):
    U, S, V = cp.linalg.svd(N, full_matrices=False)
    return cp.matmul(U[:, 0:r], cp.matmul(cp.diag(S[0:r]), V[0:r, :]))


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


def init_spectral(env, T, transition_matrices_before):
    n, S, A, H = env.n, env.S, env.A, env.H

    # Collect trimmed, low-rank approx, empirical transition matrices
    transition_matrices = []
    for a in range(A):
        # Collect empirical transition matrices
        transition_matrix_a = transition_matrices_before[a]
        # Trimming!
        ratio = (T * H) / (n * A)
        num_trimmed = int(cp.floor(n * cp.exp(- ratio * cp.log(ratio))))
        if num_trimmed > 0:
            contexts_ordered = cp.argsort(cp.sum(transition_matrix_a, axis=1))
            contexts_trimmed = contexts_ordered[-num_trimmed:]
            transition_matrix_a[contexts_trimmed, contexts_trimmed] = 0

        # Low-rank approximation
        transition_matrices.append(low_rank(transition_matrix_a, r=S))

    M_in = cp.concatenate(tuple(transition_matrices), axis=1)
    M_out = cp.concatenate(tuple(transition_matrices), axis=0).T
    M = cp.concatenate((M_in, M_out), axis=1)

    # l1-normalize rows
    row_sums = M.sum(axis=1)
    row_sums[row_sums == 0] = 1
    M = M / row_sums[:, cp.newaxis]
    # M = normalize(M, norm='l1', axis=1)

    # S-median clustering to the rows
    initial_medians = M[:S, :]
    # initial_medians = cp.random.randn(S, 2*n*A)

    # numpy + pyclustering
    if platform == "darwin":
        kmedians_instance = kmedians(M, initial_medians)
        kmedians_instance.process()
        clusters = kmedians_instance.get_clusters()
    # cupy + manual Implementation
    else:
        clusters = weighted_kmedians(M)

    f_1 = {}
    for x in range(n):
        for s in range(S):
            if x in clusters[s]:
                f_1[x] = s

    return f_1


def likelihood_improvement(env, transition_matrices_before, f_1, f, num_iter=None):
    # likelihood_improvement
    n, S, A, H = env.n, env.S, env.A, env.H

    f_final = f_1
    if num_iter is None:
        num_iter = int(cp.floor(cp.log(n * A)))
    errors = []
    fs = []

    for _ in range(num_iter):
        # create empirical latent transition matrices
        Ns = empirical_latent_transitions(env, transition_matrices_before, f_final)

        # likelihood improvement
        f_ = {}
        for x in range(n):
            likelihoods = []
            for j in range(S):
                # N2 = number of visitations to j
                N2 = 0
                for a in range(A):
                    tmp = cp.sum(Ns[a], axis=0)
                    N2 += tmp[j]
                likelihood = 0
                for a in range(A):
                    # N1 = number of visitations from (j, a)
                    tmp = cp.sum(Ns[a], axis=1)
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
                            N3 = count_transition_mixed1(x, s, transition_matrices_before[a], f)
                            # number of visitations (s, a) -> x
                            N4 = count_transition_mixed2(s, x, transition_matrices_before[a], f)

                            # compute likelihood
                            likelihood += (N3 * cp.log(p_estimated)) + (N4 * cp.log(p_bwd_estimated))
                likelihoods.append(likelihood)
            # new cluster
            # print(likelihoods)
            f_[x] = cp.argmax(likelihoods)
            fs.append(f_)
        f_final = f_
        errors.append(error_rate(f, f_final, env.n, env.S))

    return f_final, errors


def weighted_kmedians(M):
    clusters = None
    return clusters


if __name__ == '__main__':
    n = 100
    A = 2
    H = 100
    T = int(cp.ceil(n * A * (cp.log(n * A) ** 1.1)) / H)

    env_config = {'n': n, 'H': H, 'S:': 2, 'A': A,
                  'ps': [cp.array([[3 / 4, 1 / 4], [1 / 4, 3 / 4]]), cp.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])],
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
    # obtain transition matrices
    transition_matrices = empirical_transitions(env, trajectories)

    # initial spectral clustering
    f_1 = init_spectral(env, T, transition_matrices)
    init_err_rate = error_rate(f, f_1, env.n, env.S)
    print("Error rate after initial clustering is ", init_err_rate)

    # likelihood_improvement
    # f_final, errors = likelihood_improvement(env, trajectories, f_1, f, num_iter=10)
    f_final, errors = likelihood_improvement(env, transition_matrices, f_1, f, num_iter=None)
    print("Final error rate is ", errors[-1])
    print("Errors along the improvement steps: ", errors)
