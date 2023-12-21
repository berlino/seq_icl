
## Setup

```bash
conda create -n seq_icl python=3.11
pip install -r requirements.txt
```

Note that python version is 3.11.

Setup mamba-ssm with the following command:
```bash
pip install mamba-ssm # w. cuda12.1
```
And set up conv1d following the command in [this issue](https://github.com/state-spaces/mamba/issues/55)
```
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal_conv1d
git checkout v1.0.2  # this is the highest compatible version allowed by Mamba
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
```

### Experiments on Associative Recall

```bash
python -m train experiment=synthetics/associative_recall/transformer
python -m train experiment=synthetics/associative_recall/s4d
python -m train experiment=synthetics/associative_recall/h3
python -m train experiment=synthetics/associative_recall/gilr
python -m train experiment=synthetics/associative_recall/lru
python -m train experiment=synthetics/associative_recall/lstm
python -m train experiment=synthetics/associative_recall/rwkv
```

| | Transformer | S4D | H3 | GILR | LRU | LSTM | RWKV | Random
|---|:---:|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| Test Accuracy |  100.0 | 32.4 | 98.2  | 50.2  | 11.6 | 36.2 | 30.0 | 25.0 |

### Experiments on Induction Head

```bash
python -m train experiment=synthetics/induction_head/transformer
python -m train experiment=synthetics/induction_head/s4d
python -m train experiment=synthetics/induction_head/h3
python -m train experiment=synthetics/induction_head/gilr
python -m train experiment=synthetics/induction_head/lru
python -m train experiment=synthetics/induction_head/lstm
python -m train experiment=synthetics/induction_head/rwkv
```

| | Transformer | S4D | H3 | GILR | LRU | LSTM | RWKV | Random |
|---|:---:|:---:|:---:|:---:|:---:|:---:| :---:|:---:|
| Test Accuracy | 97.2 | 8.8  | 100.0   | 6.2  | 4.8 | 4.8 | 6.0 | 5.0 |

### Experiments on DFA

To run the training,
```bash
python -m train experiment=dfa/lstm
python -m train experiment=dfa/retnet
python -m train experiment=dfa/transformer+
```


To run the generation,
```bash
python -m generate experiment=dfa/lstm train.ckpt="outputs/2023-09-18/08-23-56-668022/seq-icl-data/mbg9ohwc/checkpoints/epoch\=70-step\=11147.ckpt" hydra.run.dir="./"
```
Note to escape the `=` in the checkpoint path.

To do hyperparamter sweep,

```bash
python -m sweep
```
The model family can be specified in `sweep.py`.

## Notes

* The MHA in simple\_lm.py use `num_heads`, but in other modules we use `n_heads`. The name needs to be changed for consistency, but they're kept as is for now.

### References of Linear RNNs

* [GILR](https://arxiv.org/abs/1709.04057)
* [LRU](https://arxiv.org/abs/2303.06349)

### Troubleshooting

* add `export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin` so that ldconfig can work properly

### Acknowledgements

This repo is adapted from [safari](https://github.com/HazyResearch/safari/tree/main). Triton implementations are taken from [linear rnn](https://github.com/sustcsonglin/pytorch_linear_rnn).