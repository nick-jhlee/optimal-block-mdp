"""
Created on 9/23/22
@author: nicklee

(Description)
"""
from Experiment import *

num_repeats = 100
verbose = False

delta2_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# delta2_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

n = 100
# range of T
T = 30
H = 100

# number of actions, clusters
S, A = 2, 3
# symmetric binary case
eps = 0.35
latent_transitions = [cp.array([[1 / 2 - eps, 1 / 2 + eps], [1 / 2 + eps, 1 / 2 - eps]]),
                      cp.array([[1 / 2 + eps, 1 / 2 - eps], [1 / 2 - eps, 1 / 2 + eps]]),
                      cp.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])]
# set environment
env_config = {'n': n, 'H': H, 'S:': S, 'A': A,
              'ps': latent_transitions,
              'qs': 'uniform'}
env = Synthetic(env_config)

delta1, delta3 = 0.7, 0.2

dfs = []
# main loop
for i, delta2 in enumerate(delta2_range):
    print(f"#### delta2 = {delta2} ###")

    init_errors, final_errors = simulate(env, T, num_repeats, delta1=delta1, delta2=delta2, delta3=delta3)
    if verbose:
        print("initial: ", init_errors)
        print("final: ", final_errors)
    # logging
    dfs.append(pd.DataFrame(cp.array([init_errors, final_errors]).T,
                            columns=["Init Spectral", "Likelihood Improvement"]).assign(delta2=delta2))

plot(dfs, "delta2", exp_num=5, title_str=f"Experiment 5: Varying delta2 (delta1={delta1}, delta3={delta3})")
