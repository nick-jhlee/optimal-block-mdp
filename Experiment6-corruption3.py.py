"""
Created on 9/23/22
@author: nicklee

(Description)
"""
from Experiment import *

num_repeats = 100
verbose = False

delta3_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# delta3_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

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

delta1, delta2 = 0.7, 0.2

dfs = []
# main loop
for i, delta3 in enumerate(delta3_range):
    print(f"#### delta3 = {delta3} ###")

    init_errors, final_errors = simulate(env, T, num_repeats, delta1=delta1, delta2=delta2, delta3=delta3)
    if verbose:
        print("initial: ", init_errors)
        print("final: ", final_errors)
    # logging
    dfs.append(pd.DataFrame(cp.array([init_errors, final_errors]).T,
                            columns=["Init Spectral", "Likelihood Improvement"]).assign(delta3=delta3))

plot(dfs, "delta3", exp_num=6, title_str=f"Experiment 6: Varying delta3 (delta1={delta1}, delta2={delta2})")
