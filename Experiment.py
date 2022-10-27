from Clustering import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def simulate(env, T, num_repeats, delta1=0, delta2=0, delta3=0):
    init_errors, final_errors = [], []
    for _ in range(num_repeats):
        # obtain (possibly corrupted) trajectories
        trajectories = corrupt(env, generate_trajectories(T, env), delta1=delta1, delta2=delta2, delta3=delta3)
        # obtain transition matrices
        transition_matrices = empirical_transitions(env, trajectories)

        # initial spectral clustering
        f_1 = init_spectral(env, T, transition_matrices)
        init_err_rate = error_rate(env, f_1)
        init_errors.append(init_err_rate)
        # print("Error rate after initial clustering is ", init_err_rate)

        # likelihood_improvement
        # f_final, errors = likelihood_improvement(env, trajectories, f_1, num_iter=10)
        f_final, errors = likelihood_improvement(env, transition_matrices, f_1, num_iter=None)
        final_errors.append(errors[-1])
        # print("Final error rate is ", errors[-1])
        # print("Errors along the improvement steps: ", errors)
    return init_errors, final_errors


def plot(dfs, id_var, exp_num, title_str):
    # plot and save
    cdf = pd.concat(dfs)
    mdf = pd.melt(cdf, id_vars=[id_var], var_name="Algorithm", value_name="error rate")

    fig, ax = plt.subplots()
    sns.boxplot(x=id_var, y="error rate", hue="Algorithm", data=mdf, ax=ax)
    # sns.stripplot(x="H", y="error rate", hue="Algorithm", data=mdf, dodge=True, ax=ax, legend=False)
    fig.suptitle(title_str)
    plt.savefig(f"results/plot_exp{exp_num}.pdf", dpi=500)

    plt.show()
    plt.clf()
    plt.close()

    cp.savez_compressed(f"raw_datas/exp{exp_num}", dfs=dfs)
    return None
