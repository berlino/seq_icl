# In-context Language Learning: Architectures and Algorithms [WIP]

This repo serves for the experiments for the paper:

Title: [In-context Language Learning: Architectures and Algorithms](https://arxiv.org/abs/2401.12973)

Authors : Ekin Akyürek, Bailin Wang, Yoon Kim, Jacob Andreas


## Setup

```bash
conda create -n seq_icl python=3.11
pip install -r requirements.txt
```

## Experiments


### Experiments on DFA

To run the training,
```bash
python -m train experiment=dfa/lstm
python -m train experiment=dfa/retnet
python -m train experiment=dfa/gla
python -m train experiment=dfa/transformer+
```

### Troubleshooting

* add `export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin` so that ldconfig can work properly
* The MHA in simple\_lm.py use `num_heads`, but in other modules we use `n_heads`. The name needs to be changed for consistency, but they're kept as is for now.
* you might need to set up conv1d following the command in [this issue](https://github.com/state-spaces/mamba/issues/55)
```
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal_conv1d
git checkout v1.0.2  # this is the highest compatible version allowed by Mamba
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
```

### Experiments for DeltaNet

* the fused_chunk kernel does not work for some reason
* adding a model requires adding a layer, a model, an experiment and a sweep file.

### Acknowledgements

This repo is adapted from [safari](https://github.com/HazyResearch/safari/tree/main). Triton implementations are taken from [linear rnn](https://github.com/sustcsonglin/pytorch_linear_rnn).
