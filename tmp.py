"""
Created on 10/05/22
@author: nicklee

(Description)
"""
from Clustering import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

num_repeats = 100
verbose = True

# range of n, number of contexts
n_range = [10, 100, 500, 1000, 5000, 10000]
N = len(n_range)
us = [1]
# number of actions, clusters
S, A = 2, 3
# symmetric binary case
eps = 0.4
latent_transitions = [cp.array([[1 / 2 - eps, 1 / 2 + eps], [1 / 2 + eps, 1 / 2 - eps]]),
                      cp.array([[1 / 2 + eps, 1 / 2 - eps], [1 / 2 - eps, 1 / 2 + eps]]),
                      cp.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])]

# main loop 1
for i, u in enumerate(us):
    dfs = []
    # main loop 2
    for j, n in enumerate(n_range):
        print(f"#### n = {n}, TH = 2 * n (log n)^{u} ###")
        # set H
        H = n
        # set environment
        env_config = {'n': n, 'H': H, 'S:': S, 'A': A,
                      'ps': latent_transitions,
                      'qs': 'uniform'}
        env = Synthetic(env_config)
        # true clusters
        f = {}
        for s in range(env.S):
            cluster = env.partitions[s]
            for x in range(cluster.start, cluster.start + cluster.n):
                f[x] = s

        # set T
        T = int(5 * cp.ceil(n * A * (cp.log(n * A) ** u)) / H)
        init_errors, final_errors = [], []
        for _ in range(num_repeats):
            # obtain trajectories
            trajectories = generate_trajectories(T, env)
            # obtain transition matrices
            transition_matrices = empirical_transitions(env, trajectories)

            # initial spectral clustering
            f_1 = init_spectral(env, T, transition_matrices)
            init_err_rate = error_rate(f, f_1, env.n, env.S)
            init_errors.append(init_err_rate)
            # print("Error rate after initial clustering is ", init_err_rate)

        if verbose:
            print("initial: ", init_errors)
        # logging
        dfs.append(pd.DataFrame(cp.array([init_errors]).T, columns=["Init Spectral"]).assign(n=n))

    # plot and save
    cdf = pd.concat(dfs)
    mdf = pd.melt(cdf, id_vars=['n'], var_name="Algorithm", value_name="error rate")
    # print(mdf)

    fig, ax = plt.subplots()
    sns.boxplot(x="n", y="error rate", hue="Algorithm", data=mdf, ax=ax)
    sns.stripplot(x="n", y="error rate", hue="Algorithm", data=mdf, dodge=True, ax=ax)
    fig.suptitle(f"Experiment 1: Varying n (TH = n (log n)^{u}, eps={eps})")
    plt.savefig(f"results/plot_exp1_u={u}.pdf", dpi=500)

    plt.show()
    plt.clf()
    plt.close()

    cp.savez_compressed(f"raw_datas/exp1_u={u}", dfs=dfs)
