# KDDCup2020_AutoGraph

## Introduction
My first journey to KDD Cup, and the ranks of KDD Cup 2020 AutoGraph Challenge are FeedBack 2nd and Final 10th, respectively. Moreover, the result of final phase is not so good because the model is overfitted and I do not add parameterOptimization and validation, worrying about time budget.

## Model
* PyG: Most of examples and test in PyG have been tried.
* PPNP: The version of PyG, DGL and author's, all of these are tried and the last is best. The author's code has some errors and I fix it up.
* DropEdge: I have a try, but the result is terrible, waiting the version of Pyg to validate.

## cite
The majority of my work is based on the PyG and the author of PPNP, Thanks a lot.
'''python
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
'''
