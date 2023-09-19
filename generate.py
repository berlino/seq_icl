import copy
import sys
import os
import random
import time
from functools import partial, wraps
from typing import Callable, List, Sequence

import torch
import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.utils import registry
from src.dataloaders import SequenceDataset 

log = src.utils.train.get_logger(__name__)

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

def generate(model, x, sample_len, vocab):
    """
    Args:
        x: 1d tensor of shape (seq_len,)
    """
    x = x.unsqueeze(0)
    sample = []
    for _ in range(sample_len):
        output, _ = model(x)
        logits = output.logits
        next_token_id = logits[0, -1].argmax(dim=0)
        next_token = vocab.get_vocab(next_token_id.item())
        x = torch.cat([x, next_token_id[None, None]], dim=1)
        sample.append(next_token)
    return sample


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    config = utils.train.process_config(config)
    model = utils.instantiate(registry.model, config.model)

    dataset = SequenceDataset.registry[config.dataset._name_](**config.dataset)
    dataset.setup()
    test_dataset = dataset.dataset["test"]

    ckpt_path = config.train.ckpt
    with open(ckpt_path, 'rb') as f:
        print(f'Loading from {ckpt_path}')
        state_dict = torch.load(f, map_location='cpu')["state_dict"]
        state_dict ={k.partition('model.')[2]: v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)
    
    samples = []
    for test_instance in test_dataset:
        x, y, dfa = test_instance
        separator_token = dataset.vocab.get_id("|")
        # find the index of the first padding zero in tensor x
        padding_token = 0
        first_padding_token_id = x.size(0) - (x == padding_token).sum()
        assert x[first_padding_token_id] == 0
        new_x = x[:first_padding_token_id + 1]
        new_x[first_padding_token_id] = separator_token

        sample = generate(model, new_x, 12, dataset.vocab)
        samples.append(sample)

        # TODO: evaluate if the sample belongs to the DFA
        print(sample)

if __name__ == '__main__':
    main()
