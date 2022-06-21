# Libraries and versions
python 3.6.7, numpy 1.14.3, scipy 1.1.0, scikit-learn 0.19.1, torch 0.4.0, gym 0.10.9, matplotlib 1.5.1

# Files included in this package

Decoding.py -- The decoding-based algorithm from this work.

Environments.py -- single library for loading all environments

Experiment.py -- entrypoint for experiments

GetSlopes.py -- postprocessing script for getting slope information on log-log plot.

LockBernoulli.py -- environment implementation for Lock-Bernoulli

LockGaussian.py -- environment implementation for Lock-Gaussian

OracleQ.py -- implementation of UCB-Q-Hoeffding from Jin et al. (2018)

Params.py -- wrapper infrastructure for experiments including hyperparameter configurations

Postprocess.py -- script for postprocessing data from experiments

PlotAll.py -- plotting script for comparisons

PlotSensitivity.py -- plotting script for sensitivity heatmap

QLearning.py -- implementation of Q-learning with epsilon-greedy exploration


# Running the code

Make a directory called ./data/ from here. This will be where the
files generated by the experiment script will appear.

The entry file is Experiments.py. This file takes a number of
arguments, such as algorithm, environment, environment parameters,
number of episodes, and any algorithm hyperparameters, conducts a
simulation, and writes the running average reward into a file in the
./data directory. Please see that file for details on arguments.

Hyperparameter configurations used for sweeping are in Params.py. You
may use this as follows:

```
import Params
for s in Parameters['Lock-v0']['oracleq']:
    P = Params.Params(s)
    P.iteration = 1
    print("python3 -W ignore Experiment.py %s" % (P.get_params_string()))
```

This code snippet will print all commands for the first replicate
(iteration = 1) of OracleQ on Lock-Bernoulli.

Scripts for postprocessing and plotting data are contained in
Postprocess.py, PlotAll.py, PlotSensitivity.py and GetSlopes.py



# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.