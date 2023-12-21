import os
import yaml
from yaml import Loader

import glob
import copy

import pickle
from probe import get_results

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np

from make_video import prepare_video

from probe import get_dfa_states
from bertviz import head_view, model_view
import torch

def visualize_attention_map(example, output_folder, id=0):
    t_states = len(example["states"])
    t_hidden = len(example["hidden_outputs"][0])
    t_states = len(example["states"])
    n_layers = len(example["hidden_outputs"])
    in_states = get_dfa_states(example["input"], example["dfa"], in_states=True)
    out_states = get_dfa_states(example["input"], example["dfa"], in_states=False)
    start = np.random.randint(0, t_states)
    end = min(start + 20, t_states)
    start = 0
    end = 80
    chars = list(example["input"])[start: end]
    preds = list(example["pred"])[start: end]
    # find invalid preds
    validity = []
    for t in range(start, end):
        input = "".join(chars[:t+1])
        input = input.split("|")[-1]
        test = input + preds[t]
        valid = example["dfa"](" ".join(list(test)))
        validity.append('T' if valid else 'F')

    in_states = in_states[start: end]
    out_states = out_states[start: end]
    chars = list(zip(in_states, chars, out_states, validity, preds))
    chars = [str(text) for text in chars]
    attentions = [torch.tensor(scores[None, :, start:end, start:end]) for scores in example["attention_scores"]]
    html_head_view = head_view(attentions, chars, html_action='return')

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, f"{id}.html"), 'w') as file:
        file.write(html_head_view.data)





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="s4d")
    # parser.add_argument("--hidden_key", type=str, default="hidden_outputs")
    args = parser.parse_args()

    exp_folders_40000 = {
        "s4d": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/s4d/generations/144_test.txt",
        "rwkv": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/rwkv/generations/110_test.txt",
        "retnet": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/retention/generations/36_test.txt",
        "lstm": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/lstm/generations/174_test.txt",
        "linear_transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/linear_transformer/generations/117_test.txt",
        "hyena": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/hyena/generations/28_test.txt",
        "h3": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/h3/generations/57_test.txt",
        "transformer/1": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer_1/generations/133_test.txt",
        "transformer/2": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer_2/generations/177_test.txt",
        "transformer/12": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer/generations/184_test.txt",
        "transformer/4": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer_4/generations/194_test.txt",
        "transformer/8": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer_8_w_hiddens/generations/174_test.txt",
    }

    exp_folders_2500 = {
        "s4d": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/s4d/generations/54_test.txt",
        "rwkv": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/rwkv/generations/20_test.txt",
        "retnet": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/retention/generations/106_test.txt",
        "lstm": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/lstm/generations/142_test.txt",
        "linear_transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/linear_transformer/generations/40_test.txt",
        "hyena": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/hyena/generations/59_test.txt",
        "h3": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/h3/generations/47_test.txt",
        "transformer/8": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer/generations/176_test.txt",
        "transformer/2": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer_2/generations/197_test.txt",
        "transformer/4": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer_4/generations/155_test.txt",
        "transformer/1": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer_1/generations/13_val.txt"
        # "transformer/2": "/raid/lingo/akyurek/git/iclmodels/experiments/
        # "transformer/12": "/raid/lingo/akyurek/git/iclmodels/experiments/
        # "transformer/4": "/raid/lingo/akyurek/git/iclmodels/experiments/

    }

    exp_folders_5000 = {
        "transformer/8": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_5000/transformer/generations/194_test.txt"
    }



    exp_folder = exp_folders_40000[args.exp]

    results = get_results(
        exp_folder, subset="test"
    )
    exp_folder_main = os.path.dirname(exp_folder)
    output_folder = os.path.join(exp_folder_main, "attentions")
    os.makedirs(output_folder, exist_ok=True)

    for i in range(5):
        example = results[i]
        visualize_attention_map(example, output_folder, id=i)

