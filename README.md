# Expressive Power of Invariant and Equivariant Graph Neural Networks

This repository aims at using powerful GNN (2-FGNN) on two different problems: graph alignment problem and node clustering.

We first show how to use powerful GNN (2-FGNN) to solve a graph alignment problem. This code was used to derive the practical results in the following paper:

Waiss Azizian, Marc Lelarge. Expressive Power of Invariant and Equivariant Graph Neural Networks, ICLR 2021.

[arXiv](https://arxiv.org/abs/2006.15646) [OpenReview](https://openreview.net/forum?id=lxHgXYN4bwl)

## Problem: alignment of graphs

The graph isomorphism problem is the computational problem of determining whether two finite graphs are isomorphic. Here we consider a noisy version of this problem: the two graphs below are noisy versions of a parent graph. There is no strict isomorphism between them. Can we still match the vertices of graph 1 with the corresponding vertices of graph 2?

|          graph 1          |          graph 2          |
| :-----------------------: | :-----------------------: |
| ![](images/01_graph1.png) | ![](images/02_graph2.png) |

With our GNN, we obtain the following results: green vertices are well paired vertices and red vertices are errors. Both graphs are now represented using the layout from the right above but the color of the vertices are the same on both sides. At inference, our GNN builds node embedding for the vertices of graphs 1 and 2. Finally a node of graph 1 is matched to its most similar node of graph 2 in this embedding space.

|             graph 1              |             graph 2              |
| :------------------------------: | :------------------------------: |
| ![](images/04_result_graph1.png) | ![](images/03_result_graph2.png) |

Below, on the left, we plot the errors made by our GNN: errors made on red vertices are represented by links corresponding to a wrong matching or cycle; on the right, we superpose the two graphs: green edges are in both graphs (they correspond to the parent graph), orange edges are in graph 1 only and blue edges are in graph 2 only. We clearly see the impact of the noisy edges (orange and blue) as each red vertex (corresponding to an error) is connected to such edges (except the isolated red vertex).

|  Wrong matchings/cycles  | Superposing the 2 graphs  |
| :----------------------: | :-----------------------: |
| ![](images/09_preds.png) | ![](images/05_result.png) |

To measure the performance of our GNN, instead of looking at vertices, we can look at edges. On the left below, we see that our GNN recovers most of the green edges present in graphs 1 and 2 (edges from the parent graph). On the right, mismatched edges correspond mostly to noisy (orange and blue) edges (present in only one of the graphs 1 or 2).

|      Matched edges       |      Mismatched edges       |
| :----------------------: | :-------------------------: |
| ![](images/07_match.png) | ![](images/08_mismatch.png) |

## Training GNN for the graph alignment problem

For the training of our GNN, we generate synthetic datasets as follows: first sample the parent graph and then add edges to construct graphs 1 and 2. We obtain a dataset made of pairs of graphs for which we know the true matching of vertices. We then use a siamese encoder as shown below where the same GNN (i.e. shared weights) is used for both graphs. The node embeddings constructed for each graph are then used to predict the corresponding permutation index by taking the outer product and a softmax along each row. The GNN is trained with a standard cross-entropy loss.
At inference, we can add a LAP solver to get a permutation from the matrix <img src="https://render.githubusercontent.com/render/math?math=E_1 E_2^T">.

![](images/siamese.png)

Various architectures can be used for the GNN and we find that FGNN (first introduced by Maron et al. in [Provably Powerful Graph Networks](https://papers.nips.cc/paper/2019/hash/bb04af0f7ecaee4aae62035497da1387-Abstract.html) NeurIPS 2019) are best performing for our task. In our paper [Expressive Power of Invariant and Equivariant Graph Neural Networks](https://openreview.net/forum?id=lxHgXYN4bwl), we substantiate these empirical findings by **proving that FGNN has a better power of approximation among all equivariant architectures working with tensors of order 2 presented so far** (this includes message passing GNN or linear GNN).

## Results

![](images/download.png)

Each line corresponds to a model trained at a given noise level and shows
its accuracy across all noise levels. We see that pretrained models generalize very well at noise levels unseen during the training.

We provide a simple [notebook](https://github.com/mlelarge/graph_neural_net/blob/master/plot_accuracy_regular.ipynb) to reproduce this result for the pretrained model released with this repository (to run the notebook create a `ipykernel` with name gnn and with the required dependencies as described below).

We refer to our [paper](https://openreview.net/forum?id=lxHgXYN4bwl) for comparisons with other algorithms (message passing GNN, spectral or SDP algorithms).

To cite our paper:

```
@inproceedings{azizian2020characterizing,
  title={Expressive power of invariant and equivariant graph neural networks},
  author={Azizian, Wa{\"\i}ss and Lelarge, Marc},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=lxHgXYN4bwl}
}
```

## Problem: node clustering

The node clustering problem is the computational problem of determining the cluster associated to each node in a graph.
Here we generate datasets for this problem by first sampling two graphs and then joining them. The goal is then, given the adjacency matrix, to deduce the original cluster.

![](images/graph_clustering_problem.png)

## Training GNN for the node clustering problem

For the training of our GNN, we generate synthetic datasets as follows: first sample two graphs using the Erdos Renyi algorithm with a probability p_intra. Then we join the two graphs by adding with a probability p_inter each edge between the nodes of the two graphs. We obtain then a dataset of graphs for which we know the original clusters. We then use a node embedder E as shown below and compute a similarity matrix EE^T. The GNN is trained to maximize the similarity between nodes from the same cluster, and to minimize the similarity between nodes from different clusters.
In order to mesure the accuracy of the model, we can use a k-means algorithm to get clusters from EE^T and then compare them with the original ones.

![](images/similarity_net.png)

## Hyperparameter optimization

![](images/acc_hyperparam.png)

## Results

Firstly, we trained FGNNs on specific probability pairs (p_intra, p_inter). We found that, as shown in the image below, with the current hyperparameters in `default_cluster.yaml`, the model is not overfitting our training sets.

![](images/cluster__edge_prob__intra_0.80_inter_0.60.png)

Then we trained FGNNs on specific probability pairs (p_intra, p_inter) generated randomly and plotted their associated validation accuracy in a grid shown below. The model seems very accurate for each probabilities levels, except when the intra and inter probabilities are very close. In order to measure the efficiency of our model, we also plotted the results of the spectral clustering algorithm on our datasets. We found that our model can be as efficient as the spectral algorithm on every probability pairs such that p_inter <= p_intra.

|                FGNN                 |     Spectral clustering      |
| :---------------------------------: | :--------------------------: |
| ![](images/acc_edge_prob_ep_14.png) | ![](images/acc_spectral.png) |

However, when we tried to train one FGNN on every pairs, we found that, even if the model has quite good results, the accuracy were not as good as FGNNs trained on specific pairs or the spectral clustering algorithm.

![](images/acc_global_edge_prob.png)

We then tried to evaluate the ability of the model to generalize. We trained one FGNN with p_inter=0.9 and p_intra=0.7, and tested its accuracy on different pairs:

![](images/acc_0.9_0.7.png)

The model seems able to generalize for probability pairs near the one it was trained and surprisingly on pairs which are symmetric to the ones where it generalizes quite well. However it performs very poorly on pairs that are far from the one it was trained on.

In the image below, we show a 2D representation of the nodes embedding deduced from the similarity matrix EE^T. The clusters were generated using Erdos Renyi algorithm with p_intra=0.5. The clusters were then joined with a probability p_inter=0.2 for each pair of nodes from the two clusters.
![](images/clustering_sample_result.png)

## Overview of the code

### Project structure

```bash
.
├── loaders
|   └── dataset selector
|   └── data_generator.py # generating random graphs
|   └── data_generator_label.py # generating random 2-cluster graphs
|   └── test_data_generator.py
|   └── siamese_loader.py # loading pairs
├── models
|   └── architecture selector
|   └── layers.py # equivariant block
|   └── base_model.py # powerful GNN Graph -> Graph
|   └── siamese_net.py # GNN to match graphs
|   └── similarity_net.py # GNN to classifye nodes of a graph
├── toolbox
|   └── optimizer and losses selectors
|   └── logger.py  # keeping track of most results during training
|   └── metrics.py # computing scores
|   └── losses.py  # computing losses
|   └── optimizer.py # optimizers
|   └── utility.py
|   └── maskedtensor.py # Tensor-like class to handle batches of graphs of different sizes
├── commander.py # main file from the project of graph alignment serving for calling all necessary functions for training and testing
├── commander_label.py # main file from the project of nodes classification serving for calling all necessary functions for training and testing
├── trainer.py # pipelines for training and validation
├── eval.py # testing models
```

## Dependencies

Dependencies are listed in `requirements.txt`. To install, run

```
pip install -r requirements.txt
```

## Training

Run the main file `commander.py` with the command `train` for the graph alignment problem. For the node clustering, replace `commander.py` by `commander_label.py`.

```
python commander.py train
```

To change options, use [Sacred](https://github.com/IDSIA/sacred) command-line interface and see `default.yaml` for the configuration structure. For instance,

```
python commander.py train with cpu=No data.generative_model=Regular train.epoch=10
```

You can also copy `default.yaml` and modify the configuration parameters there. Loading the configuration in `other.yaml` (or `other.json`) can be done with

```
python commander.py train with other.yaml
```

See [Sacred documentation](http://sacred.readthedocs.org/) for an exhaustive reference.

To save logs to [Neptune](https://neptune.ai/), you need to provide your own API key via the dedicated environment variable.

The model is regularly saved in the folder `runs`.

## Evaluating

There are two ways of evaluating the models. If you juste ran the training with a configuration `conf.yaml`, you can simply do,

```
python commander.py eval with conf.yaml
```

You can omit `with conf.yaml` if you are using the default configuration.

If you downloaded a model with a config file from here, you can edit the section `test_data` of this config if you wish and then run,

```
python commander.py eval with /path/to/config model_path=/path/to/model.pth.tar
```

## Multiprocessing

If you have SLURM installed, you can run multiple configuration files with `generate_configs.py`. The method `runConfigs(n, pythonScript, cmd)` from this file will execute every configuration file in the folder `configs/configs_to_run` on the number of machines specified. The logs are then stored in the folder `configs/configs_computed`. You can also generate configuration files with `generate_configs.py` and then run them with `runConfigs`.
