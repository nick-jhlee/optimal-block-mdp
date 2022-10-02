"""
Created on 9/23/22
@author: nicklee

(Description)
"""
from Clustering import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

num_repeats = 20
verbose = False

n = 100
# set H
# H = int(np.ceil(n / 2))
H = n
# range of T
T_range = cp.array(range(5, 50, 5))
T_num = len(T_range)

# number of actions, clusters
S, A = 2, 3
# symmetric binary case
eps = 0.3
latent_transitions = [cp.array([[1 / 2 - eps, 1 / 2 + eps], [1 / 2 + eps, 1 / 2 - eps]]),
                      cp.array([[1 / 2 + eps, 1 / 2 - eps], [1 / 2 - eps, 1 / 2 + eps]]),
                      cp.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])]

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

dfs = []
# main loop
for i, T in enumerate(T_range):
    print(f"#### TH = {T * H} ###")
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

        # likelihood_improvement
        # f_final, errors = likelihood_improvement(env, trajectories, f_1, f, num_iter=10)
        f_final, errors = likelihood_improvement(env, transition_matrices, f_1, f, num_iter=None)
        final_errors.append(errors[-1])
        # print("Final error rate is ", errors[-1])
        # print("Errors along the improvement steps: ", errors)
    if verbose:
        print("initial: ", init_errors)
        print("final: ", final_errors)
    # logging
    dfs.append(pd.DataFrame(cp.array([init_errors, final_errors]).T, columns=["Init Spectral", "Likelihood Improvement"]).assign(T=T))

# plot and save
cdf = pd.concat(dfs)
mdf = pd.melt(cdf, id_vars=['T'], var_name="Algorithm", value_name="error rate")
# print(mdf)

fig, ax = plt.subplots()
sns.boxplot(x="T", y="error rate", hue="Algorithm", data=mdf, ax=ax)
sns.stripplot(x="T", y="error rate", hue="Algorithm", data=mdf, dodge=True, ax=ax)
fig.suptitle(f"Experiment 2: Varying TH (n={n}, H={H}, eps={eps})")
plt.savefig("results/plot_exp2.pdf", dpi=500)

plt.show()
plt.clf()
plt.close()

cp.savez_compressed("raw_datas/exp2", dfs=dfs)