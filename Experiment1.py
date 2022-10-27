"""
Created on 9/23/22
@author: nicklee

(Description)
"""
from Experiment import *

num_repeats = 100
verbose = False

n = 100
# set H
# H = int(np.ceil(n / 2))
H_range = range(int(n / 5), n + 10, 10)
# range of T
T = 20

# number of actions, clusters
S, A = 2, 3
# symmetric binary case
eps = 0.3
latent_transitions = [cp.array([[1 / 2 - eps, 1 / 2 + eps], [1 / 2 + eps, 1 / 2 - eps]]),
                      cp.array([[1 / 2 + eps, 1 / 2 - eps], [1 / 2 - eps, 1 / 2 + eps]]),
                      cp.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])]

dfs = []
# main loop
for i, H in enumerate(H_range):
    print(f"#### TH = {T * H} ###")
    # set environment
    env_config = {'n': n, 'H': H, 'S:': S, 'A': A,
                  'ps': latent_transitions,
                  'qs': 'uniform'}
    env = Synthetic(env_config)

    init_errors, final_errors = simulate(env, T, num_repeats)
    if verbose:
        print("initial: ", init_errors)
        print("final: ", final_errors)
    # logging
    dfs.append(pd.DataFrame(cp.array([init_errors, final_errors]).T,
                            columns=["Init Spectral", "Likelihood Improvement"]).assign(H=H))

plot(dfs, "H", exp_num=1, title_str=f"Experiment 1: Varying H (n={n}, T={T}, eps={eps})")
