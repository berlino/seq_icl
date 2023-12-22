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

import matplotlib.cm as cm


def plot_clusters(
    results,
    folder,
    indices=range(2),
    hidden_key="hidden_outputs",
    reduction="tsne",
    layer_wise=False,
    factors=None,
    fill_types=None,
    colors=None,
):
    assert len(results) == 500, f"len(results)={len(results)}"

    for i in indices:
        example = copy.deepcopy(results[i])
        # offset
        # example[hidden_key] = [h[300:] for h in example[hidden_key]]
        # example["states"] = example["states"][300:]
        # example["input"] = example["input"][300:]

        # t_hidden = len(example[hidden_key][0])
        t_states = len(example["states"])
        # take hidden states upto t_states
        example[hidden_key] = [h[:t_states].reshape(t_states, -1) for h in example[hidden_key]]

        n_layers = len(example[hidden_key])
        alphabet = sorted(set(example["input"]).difference({"|"}))
        states = sorted(set(example["states"]).difference({"-1"}))
        reducer = TSNE if reduction == "tsne" else PCA
        # breakpoint()

        if layer_wise:
            Xs = [
                reducer(n_components=2).fit_transform(hidden_outputs)
                for hidden_outputs in example[hidden_key]
            ]
            X = np.concatenate(Xs, axis=0)
            X = X.reshape(n_layers, t_states, 2)
        else:
            hidden_outputs = np.concatenate(example[hidden_key], axis=0)
            X = reducer(n_components=2).fit_transform(hidden_outputs)
            X = X.reshape(n_layers, t_states, 2)

#         X = X[:, :t_states, :]

        for t in range(20, t_states, 10):
            a, _ = factors[n_layers]
            axes, fig = plt.subplots(a, a, figsize=(16, 16))
            axes.suptitle(
                f"T={t}, #states: "
                + str(len(example["dfa"].dfa._transition_function))
                + ", len(vocab): "
                + str(len(alphabet))
            )

            for layer in range(n_layers - 1, -1, -1):
                X_layer = X[layer, :t]
                labels = np.array(example["states"][:t])
                chars = np.array(list(example["input"][:t]))
                # vocab = example["vocab"][1:]
                ax = fig[layer // a, layer % a]
                for label in states:
                    if label == -1 or label == "-1":
                        continue
                    for ci, v in enumerate(alphabet):
                        indices = np.where((labels == label) & (chars == v))
                        if len(indices) > 0:
                            ax.scatter(
                                X_layer[indices, 0],
                                X_layer[indices, 1],
                                marker=fill_types[states.index(label)],
                                c=colors[ci],
                            )
                            # show legend
                # set title
                ax.set_title(f"layer={layer}")
            plt.savefig(os.path.join(folder, f"e_{i}_{reduction}_t_{t}.jpg"))
            plt.close()

        video_out_folder = os.path.join(folder, "video")
        video_glob = folder + "/" + f"e_{i}_{reduction}_t_*.jpg"
        prepare_video(video_glob, video_out_folder, f"e_{i}_{reduction}")
        print(f"done at {video_out_folder}")

    return video_out_folder


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="s4d")
    parser.add_argument("--hidden_key", type=str, default="hidden_outputs")
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

    fill_types = (
        "o",
        "8",
        "*",
        "v",
        "X",
        "^",
        "<",
        ">",
        "s",
        "p",
        "h",
        "H",
        "D",
        "d",
        "P",
    )
    colors = cm.Set1.colors + (
        (0.0, 0.0, 0.0),
        cm.Set2.colors[0],
        cm.tab20b.colors[-1],
        cm.tab20b.colors[-2],
        cm.tab20b.colors[-3],
    )
    factors = {
        4: (2, 2),
        9: (3, 3),
        8: (3, 3),
        13: (4, 4),
        5: (3, 3),
        3: (2, 2),
        1: (1, 1),
    }

    exp_folder = exp_folders_2500[args.exp]
    print(exp_folder)
    print(args)

    results = get_results(
        exp_folder, subset="test", key=args.hidden_key, in_states=True
    )
    # video_folders = []
    # for exp_name, folder in exp_folders.items():
    exp_folder_main = os.path.dirname(exp_folder)
    plot_folder = os.path.join(exp_folder_main, "plots", args.hidden_key)
    print(plot_folder)
    # make
    os.makedirs(plot_folder, exist_ok=True)
    output_video_folder = plot_clusters(
        results,
        plot_folder,
        hidden_key=args.hidden_key,
        indices=[0,],
        reduction="pca",
        layer_wise=True,
        factors=factors,
        fill_types=fill_types,
        colors=colors,
    )
    print(output_video_folder)
