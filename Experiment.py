from Clustering import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Experiment 1. Changing n, depending on the regime of T and H

    # range of n, number of contexts
    # n_range = range(100, 500, 20)
    n_range = range(100, 500, 20)
    N = len(n_range)
    num_repeats = 5
    # us = [0, 0.5, 1, 1.5, 2]  # TH = n (log n)^u
    us = [0, 1, 2]  # TH = n (log n)^u
    # number of actions, clusters
    S, A = 2, 2
    # symmetric binary case
    eps = 1 / 6
    latent_transitions = [np.array([[1 / 2 - eps, 1 / 2 + eps], [1 / 2 + eps, 1 / 2 - eps]]),
                          np.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])]

    init_errors_mean = np.ones((len(us), len(n_range)))
    init_errors_std = np.ones((len(us), len(n_range)))
    final_errors_mean = np.ones((len(us), len(n_range)))
    final_errors_std = np.ones((len(us), len(n_range)))

    fig, axs = plt.subplots(len(us), sharex=True)

    # main loop 1
    for j, n in enumerate(n_range):
        # set H
        H = int(np.ceil(n / 2))
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
        # main loop 2
        for i, u in enumerate(us):
            print(f"\n\n#### n = {j}, TH = n (log n)^{u} ### \n")
            # set T
            T = int(np.ceil(n * A * (np.log(n * A) ** u)) / H)
            init_errors, final_errors = [], []
            for _ in range(num_repeats):
                # obtain trajectories
                trajectories = generate_trajectories(T, env)

                # initial spectral clustering
                f_1 = init_spectral(env, trajectories)
                init_err_rate = error_rate(f, f_1, env.n, env.S)
                init_errors.append(init_err_rate)
                # print("Error rate after initial clustering is ", init_err_rate)

                # likelihood_improvement
                # f_final, errors = likelihood_improvement(env, trajectories, f_1, f, num_iter=10)
                f_final, errors = likelihood_improvement(env, trajectories, f_1, f, num_iter=None)
                final_errors.append(errors[-1])
                # print("Final error rate is ", errors[-1])
                # print("Errors along the improvement steps: ", errors)

                # PLOTS
                # axs[i].scatter(n_range, init_errors[i])
                # axs[i].scatter(n_range, final_errors[i])
                # fig.savefig("plot.pdf", dpi=300)

            # logging
            init_errors_mean[i][j] = np.mean(init_errors)
            init_errors_std[i][j] = np.std(init_errors)
            final_errors_mean[i][j] = np.mean(final_errors)
            final_errors_std[i][j] = np.std(final_errors)

    # FINAL PLOTS
    fig, axs = plt.subplots(len(us), sharex=True)
    for i, u in enumerate(us):
        axs[i].errorbar(n_range, init_errors_mean[i], yerr=init_errors_std)
        axs[i].errorbar(n_range, final_errors_mean[i], yerr=final_errors_std)
    fig.savefig("plot_final.pdf", dpi=500)