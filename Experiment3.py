"""
Created on 9/26/22
@author: nicklee

(Description)
"""
from Experiment import *

num_repeats = 100
verbose = False

n = 100
# set T, H
H = int(n/2)
T = 30

# number of actions, clusters
S, A = 2, 3

# range of epsilon
eps_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
eps_num = len(eps_range)

dfs = []
# main loop
for i, eps in enumerate(eps_range):
    eps = round(eps, 2)
    # symmetric binary case
    latent_transitions = [cp.array([[1 / 2 - eps, 1 / 2 + eps], [1 / 2 + eps, 1 / 2 - eps]]),
                          cp.array([[1 / 2 + eps, 1 / 2 - eps], [1 / 2 - eps, 1 / 2 + eps]]),
                          cp.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])]

    # set environment
    env_config = {'n': n, 'H': H, 'S:': S, 'A': A,
                  'ps': latent_transitions,
                  'qs': 'uniform'}
    env = Synthetic(env_config)

    print(f"#### epsilon = {eps} ###")
    init_errors, final_errors = simulate(env, T, num_repeats)
    if verbose:
        print("initial: ", init_errors)
        print("final: ", final_errors)
    # logging
    dfs.append(pd.DataFrame(cp.array([init_errors, final_errors]).T,
                            columns=["Init Spectral", "Likelihood Improvement"]).assign(eps=eps))

plot(dfs, "eps", exp_num=3, title_str=f"Experiment 3: Varying eps (n={n}, T={T}, H={H})")
