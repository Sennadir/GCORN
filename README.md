# GCORN - Bounding the Expected Robustness of Graph Neural Networks Subject to Node Feature Attacks

This repository is the official implementation of our paper "Bounding the Expected Robustness of Graph Neural Networks Subject to Node Feature Attacks".
The paper is available online in the [OpenReview](https://openreview.net/forum?id=DfPtC8uSot).

## Requirements

Code is written in Python 3.6 and requires:

- PyTorch
- Torch Geometric
- NetworkX


## Datasets
For node classifiation, the used datasets are as follows:
- Cora
- CiteSeer
- PubMed
- CS

All these datasets are part of the torch_geometric datasets and are directly downloaded when running the code.


## Training and Evaluation

To train and evaluate the model in the paper, the user should specify the following :

- Dataset : The dataset to be used
- hidden_dimension: The hidden dimension used in the model
- learning rate and epochs
- Budget: The budget of the attack
- Type of the attack: we evaluate using Random Attack and the PGD attack.

To run a normal code of GCORN without attack for the default values with Cora dataset:

```bash
python run_gcorn.py --dataset Cora
```

## Results reproduction
To reproduce the results in the paper that compare the GCN, RGCN and the GCORN, use the following:

```bash
python main.py --dataset Cora --budget 0.5 --attack random
```


## Reproduction
For other benchmarks used in our paper (AIRGNN, GCN-k), please refer directly to their available Github repository.
For all the details related to the code and the implementation, please refer to our paper.

## Citing
If you find our proposed GCORN useful for your research, please consider citing our paper.

For any additional questions/suggestions you might have about the code and/or the proposed approach to ennadir@kth.se.

## Licence
NoisyGNN is licensed under the MIT License.
