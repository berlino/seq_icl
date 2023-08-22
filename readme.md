
### Experiments on Associative Recall

```bash
python -m train experiment=synthetics/associative_recall/transformer
python -m train experiment=synthetics/associative_recall/s4d
python -m train experiment=synthetics/associative_recall/h3
python -m train experiment=synthetics/associative_recall/gilr
python -m train experiment=synthetics/associative_recall/lru
python -m train experiment=synthetics/associative_recall/lstm
```

| | Transformer | S4D | H3 | GILR | LRU | LSTM | Random 
|---|:---:|:---:|:---:|:---:|:---:| :---:|
| Test Accuracy |  100.0 | 32.4 | 98.2  | 50.2  | 11.6 | 36.2 | 25.0 |

### Experiments on Induction Head

```bash
python -m train experiment=synthetics/induction_head/transformer
python -m train experiment=synthetics/induction_head/s4d
python -m train experiment=synthetics/induction_head/h3
python -m train experiment=synthetics/induction_head/gilr
python -m train experiment=synthetics/induction_head/lru
python -m train experiment=synthetics/induction_head/lstm
```

| | Transformer | S4D | H3 | GILR | LRU | LSTM | Random |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Test Accuracy | 97.2 | 8.8  | 100.0   | 6.2  | 4.8 | 4.8 | 5.0 |


## Notes

### References of Linear RNNs

* [GILR](https://arxiv.org/abs/1709.04057)
* [LRU](https://arxiv.org/abs/2303.06349)

### Troubleshooting

* add `export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin` so that ldconfig can work properly

### Acknowledgements

This repo is adapted from [safari](https://github.com/HazyResearch/safari/tree/main).