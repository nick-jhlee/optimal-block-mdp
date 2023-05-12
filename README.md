# optimal-block-mdp
This repository provides the codes for reproducing the experiments described in the paper [*Nearly Optimal Latent State Decoding in Block MDPs*](https://proceedings.mlr.press/v206/jedra23a.html), accepted to AISTATS 2023, by Yassir Jedra*, Junghyun Lee*, Alexandre Proutière, and Se-Young Yun.

If you plan to use this repository or refer to our work, please use the following bibtex format:

```latex
@InProceedings{jedra2023bmdp,
  title = 	 {{Nearly Optimal Latent State Decoding in Block MDPs}},
  author =       {Jedra, Yassir and Lee, Junghyun and Proutiere, Alexandre and Yun, Se-Young},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {2805--2904},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/jedra23a/jedra23a.pdf},
  url = 	 {https://proceedings.mlr.press/v206/jedra23a.html},
  abstract = 	 {We consider the problem of model estimation in episodic Block MDPs. In these MDPs, the decision maker has access to rich observations or contexts generated from a small number of latent states. We are interested in estimating the latent state decoding function (the mapping from the observations to latent states) based on data generated under a fixed behavior policy. We derive an information-theoretical lower bound on the error rate for estimating this function and present an algorithm approaching this fundamental limit. In turn, our algorithm also provides estimates of all the components of the MDP. We apply our results to the problem of learning near-optimal policies in the reward-free setting. Based on our efficient model estimation algorithm, we show that we can infer a policy converging (as the number of collected samples grows large) to the optimal policy at the best possible rate. Our analysis provides necessary and sufficient conditions under which exploiting the block structure yields improvements in the sample complexity for identifying near-optimal policies. When these conditions are met, the sample complexity in the minimax reward-free setting is improved by a multiplicative factor $n$, where $n$ is the number of possible contexts.}
}
```


## Reproducing Experiments

run main.py
