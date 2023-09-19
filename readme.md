
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
```


To run the generation, 
```bash
python -m generate experiment=dfa/lstm train.ckpt="outputs/2023-09-18/08-23-56-668022/seq-icl-data/mbg9ohwc/checkpoints/epoch\=70-step\=11147.ckpt" hydra.run.dir="./"
```
Note to escape the `=` in the checkpoint path.

## Notes

### References of Linear RNNs

* [GILR](https://arxiv.org/abs/1709.04057)
* [LRU](https://arxiv.org/abs/2303.06349)

### Troubleshooting

* add `export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin` so that ldconfig can work properly

### Acknowledgements

This repo is adapted from [safari](https://github.com/HazyResearch/safari/tree/main). Triton implementations are taken from 