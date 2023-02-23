"""
Created on 2023.02.22
@author: nicklee

(Description)
"""
from Experiment import *


def main(S, A, n, num_repeats, param_range, param_name):
    dfs = []
    delta1, delta2, delta3 = 0, 0, 0
    for i, param in enumerate(param_range):
        if param_name == 'eta':
            eta = round(param, 2)
            T = 30
            H = n

            exp_num = 3
        else:
            # symmetric binary case
            eta = 4.0
            if param_name == 'T':
                T = param
                H = n

                exp_num = 1
            elif param_name == 'H':
                T = 30
                H = param

                exp_num = 2
            elif 'delta' in param_name:
                T = 30
                H = n

                exp_num = 3 + int(param_name[-1])

                # delta1: proportion of trajectories to be corrupted
                # delta2: proportion of contexts to be corrupted, per trajectory
                # delta3: proportion of actions to be corrupted, per trajectory
                if exp_num == 4:
                    delta1 = param
                    delta2, delta3 = 0.3, 0.3
                elif exp_num == 5:
                    delta2 = param
                    delta1, delta3 = 0.3, 0.3
                elif exp_num == 6:
                    delta3 = param
                    delta1, delta2 = 0.3, 0.3
                else:
                    raise NotImplementedError(f"Experiment {exp_num} not implemented!")
            else:
                raise NotImplementedError(f"{param_name} not implemented!")

        # symmetric binary case
        latent_transitions = [cp.array([[eta / (1 + eta), 1 / (1 + eta)], [1 / (1 + eta), eta / (1 + eta)]]),
                              cp.array([[1 / (1 + eta), eta / (1 + eta)], [eta / (1 + eta), 1 / (1 + eta)]]),
                              cp.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])]
        # set environment
        env_config = {'n': n, 'H': H, 'S:': S, 'A': A,
                      'ps': latent_transitions,
                      'qs': 'uniform'}
        env = Synthetic(env_config)

        init_errors, final_errors = simulate(env, T, num_repeats, f"{param_name} = {param}", delta1, delta2, delta3)
        if verbose:
            print("initial: ", init_errors)
            print("final: ", final_errors)
        # logging
        dfs.append(pd.DataFrame(cp.array([init_errors, final_errors]).T,
                                columns=["Init Spectral", "Likelihood Improvement"]).assign(param=param))

    # save raw data
    cp.savez_compressed(f"raw_datas/exp{exp_num}", dfs=dfs)
    # plot
    plot(dfs, exp_num, f"Varying {param_name}")


if __name__ == '__main__':
    num_repeats = 1000
    verbose = False

    # number of actions, clusters
    S, A = 2, 3
    n = 100

    # Experiment 1. Varying T
    T_range = range(5, 50, 5)
    main(S, A, n, num_repeats, T_range, 'T')

    # Experiment 2. Varying H
    H_range = range(int(n / 5), n + 10, 10)
    main(S, A, n, num_repeats, H_range, 'H')

    # Experiment 3. Varying eta
    eta_range = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    main(S, A, n, num_repeats, eta_range, 'eta')

    # Experiment 4. Varying delta1
    delta1_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    main(S, A, n, num_repeats, delta1_range, 'delta1')

    # Experiment 5. Varying delta2
    delta2_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    main(S, A, n, num_repeats, delta2_range, 'delta2')

    # Experiment 6. Varying delta3
    delta3_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    main(S, A, n, num_repeats, delta3_range, 'delta3')
