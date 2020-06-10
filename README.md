[![License: MIT](https://img.shields.io/badge/license-MIT-blue](./LICENSE)
# KDDCup2020_AutoGraph

## Introduction
My first journey to KDD Cup, and the ranks of KDD Cup 2020 AutoGraph Challenge are FeedBack 2nd and Final 10th, respectively. Moreover, the result of final phase is not so good because the model is overfitted and I do not add parameterOptimization and validation, worrying about time budget.

## Model
* PyG: Most of examples and test in PyG have been tried.
* PPNP: The version of PyG, DGL and author's, all of these are tried and the last is best. The author's code has some errors and I fix it up.
* DropEdge: I have a try, but the result does not meet expectations, waiting the version of Pyg to validating.

## Inference
The majority of my work is based on the PyG and PPNP, thanks a lot.
```
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```
> Predict then Propagate: Graph Neural Networks meet Personalized PageRank. Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann. ICLR, 2019. [Paper](https://arxiv.org/abs/1810.05997)
