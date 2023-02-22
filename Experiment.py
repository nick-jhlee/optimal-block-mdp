from Clustering import *

import multiprocess
from multiprocess import Pool
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def simulate_single_run(env, T, delta1=0, delta2=0, delta3=0):
    # obtain (possibly corrupted) trajectories
    trajectories = corrupt(env, generate_trajectories(T, env), delta1=delta1, delta2=delta2, delta3=delta3)
    # obtain transition matrices
    transition_matrices = empirical_transitions(env, trajectories)

    # initial spectral clustering
    f_1 = init_spectral(env, T, transition_matrices)
    init_err_rate = error_rate(env, f_1)
    # print("Error rate after initial clustering is ", init_err_rate)

    # likelihood_improvement
    # f_final, errors = likelihood_improvement(env, trajectories, f_1, num_iter=10)
    f_final, errors = likelihood_improvement(env, transition_matrices, f_1, num_iter=None)

    # return init and final errors
    return init_err_rate, errors[-1]


# source: https://stackoverflow.com/questions/19562916/print-progress-of-pool-map-async
def track_job(job, exp_description, update_interval=3):
    while job._number_left > 0:
        tmp = str(job._number_left * job._chunksize).zfill(4)
        print(f"({exp_description}) Tasks remaining = {tmp}", end='\r')
        time.sleep(update_interval)


def simulate(env, T, num_repeats, exp_description, delta1=0, delta2=0, delta3=0):
    def F(repeat):
        result = simulate_single_run(env, T, delta1, delta2, delta3)

        # # print process id
        # print(f"Finished: {multiprocess.process.current_process()}, TH = {T * env.H}")
        return result

    repeats = list(range(num_repeats))
    # run the experiments in parallel
    with Pool() as pool:
        everything = pool.map_async(F, repeats)
        track_job(everything, exp_description)
        everything = everything.get()

    tmp = list(zip(*everything))
    return list(tmp[0]), list(tmp[1])


def plot_final_discrete(final_means, final_stds, xs, title, xlabel, legends, fname):
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", len(legends))

    with sns.axes_style("darkgrid"):
        for i, color in enumerate(clrs):
            ax.errorbar(xs, final_means[i], yerr=final_stds[i], fmt='o', linestyle='dashed', capsize=3,
                        label=legends[i], c=color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig(fname, dpi=1200, bbox_inches='tight')
    plt.show()
    plt.close()


def plot(dfs, exp_num, title_str):
    # plot and save
    cdf = pd.concat(dfs)
    mdf = pd.melt(cdf, id_vars=["param"], var_name="Algorithm", value_name="error rate")

    fig, ax = plt.subplots()
    sns.boxplot(x="param", y="error rate", hue="Algorithm", data=mdf, ax=ax, whis=1.1)
    # sns.stripplot(x="H", y="error rate", hue="Algorithm", data=mdf, dodge=True, ax=ax, legend=False)
    fig.suptitle(title_str)
    plt.savefig(f"results/plot_exp{exp_num}.pdf", dpi=500)

    # plt.show()
    plt.clf()
    plt.close()
