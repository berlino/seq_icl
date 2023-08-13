
### Experiments on Associative Recall

```bash
python -m train experiment=synthetics/associative_recall/transformer
python -m train experiment=synthetics/associative_recall/s4d
python -m train experiment=synthetics/associative_recall/h3
python -m train experiment=synthetics/associative_recall/gilr
python -m train experiment=synthetics/associative_recall/lru
```

| | Transformer | S4D | H3 | GILR | LRU |
|---|---|---|---|---|---|
| Accuracy |  |  |  |  | |

### Experiments on Induction Head

```bash
python -m train experiment=synthetics/induction_head/transformer
python -m train experiment=synthetics/induction_head/s4d
python -m train experiment=synthetics/induction_head/h3
python -m train experiment=synthetics/induction_head/gilr
python -m train experiment=synthetics/induction_head/lru
```

| | Transformer | S4D | H3 | GILR | LRU |
|---|---|---|---|---|---|
| Accuracy |  |  |  |  | |

## Acknowledgements

This repo is adapted from [safari](https://github.com/HazyResearch/safari/tree/main).