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
import seaborn as sns
import torch

import scienceplots

plt.style.use(['science','ieee'])
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['xtick.minor.visible'] = False
# increase fonts for xticks
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# increse title size
plt.rcParams['axes.titlesize'] = 14

def visualize_attention_map(example, output_folder, id=0):
    t_states = len(example["states"])
    t_hidden = len(example["hidden_outputs"][0])
    t_states = len(example["states"])
    n_layers = len(example["hidden_outputs"])
    in_states = get_dfa_states(example["input"], example["dfa"], in_states=True)
    out_states = get_dfa_states(example["input"], example["dfa"], in_states=False)
    start = np.random.randint(0, t_states)
    end = min(start + 20, t_states)
    start = 52
    end = 80
    chars = list(example["input"])[start: end]
    preds = list(example["pred"])[start: end]
    # find invalid preds
    validity = []
    for t in range(0, len(chars)):
        input = "".join(chars[:t+1])
        input = input.split("|")[-1]
        test = input + preds[t]
        valid = example["dfa"](" ".join(list(test)))
        validity.append('T' if valid else 'F')

    in_states = in_states[start: end]
    out_states = out_states[start: end]
    chars = [f"{char.replace('|', 'I')} {ostate}" for istate, char, ostate, label, pred in zip(in_states, chars, out_states, validity, preds)]
    # chars = [str(text) for text in chars]
    attentions = [torch.tensor(scores[None, :, start:end, start:end]) for scores in example["attention_scores"]]
    html_head_view = head_view(attentions, chars, html_action='return')

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, f"{id}.html"), 'w') as file:
        file.write(html_head_view.data)

    os.makedirs(os.path.join(output_folder, f"{id}"), exist_ok=True)

    for layer in range(len(example["attention_scores"])):
        attentions = example["attention_scores"][layer][:, start:end, start:end] # 2 x T X T
        for head in range(attentions.shape[0]):
            # new figure
            # new figure
            plt.figure(figsize=(10, 10))
            # mask upper triangle
            mask = np.zeros_like(attentions[head])
            mask[np.triu_indices_from(mask)] = True
            # make diagonal false
            np.fill_diagonal(mask, False)
            ax = sns.heatmap(attentions[head],
                        xticklabels=chars,
                        yticklabels=chars,
                        mask=mask,
                        cmap="Blues",
                        square=True,
                        cbar_kws={"orientation": "horizontal", "pad": 0.06, "shrink": 0.75},
                        annot=False)
            ax.set_title(f"L={layer}, H={head}")
            ax.yaxis.tick_right()
            ax.set_aspect("equal")
            plt.setp(ax.get_xticklabels(), rotation=60)
            plt.setp(ax.get_yticklabels(), rotation=0)
            path = os.path.join(output_folder, f"{id}", f"L{layer}H{head}.jpeg")
            plt.tight_layout()
            plt.savefig(path)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="transformer/12")
    parser.add_argument("--num_examples", type=str, default="40000")

    args = parser.parse_args()

    exp_folders_40000 = {
        "linear_transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/linear_transformer/generations/65_test.txt",
        "transformer/1": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer_1/generations/133_test.txt",
        "transformer/2": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer_2/generations/177_test.txt",
        "transformer/12": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer/generations/184_test.txt",
        "transformer/4": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer_4/generations/194_test.txt",
        "transformer/8": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer_8/generations/174_test.txt",
    }

    exp_folders_2500 = {
        "linear_transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/linear_transformer/generations/40_test.txt",
        "transformer/12": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer/generations/194_test.txt",
        "transformer/8": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer_8/generations/176_test.txt",
        "transformer/2": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer_2/generations/197_test.txt",
        "transformer/4": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer_4/generations/155_test.txt",
        "transformer/1": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer_1/generations/13_test.txt"

    }

    exp_folders_5000 = {
        "transformer/8": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_5000/transformer/generations/194_test.txt"
    }

    if args.num_examples == "40000":
        exp_folders = exp_folders_40000
    elif args.num_examples == "2500":
        exp_folders = exp_folders_2500
    elif args.num_examples == "5000":
        exp_folders = exp_folders_5000


    exp_folder = exp_folders[args.exp]

    results = get_results(
        exp_folder, subset="test"
    )
    exp_folder_main = os.path.dirname(exp_folder)
    output_folder = os.path.join(exp_folder_main, "attentions")
    os.makedirs(output_folder, exist_ok=True)

    for i in range(5):
        example = results[i]
        visualize_attention_map(example, output_folder, id=i)

