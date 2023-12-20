import os
import yaml
from yaml import Loader

import glob

import pickle
from probe import get_results

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np

from make_video import prepare_video

import matplotlib.cm as cm

PATHS = [
    "outputs/2023-10-18/11-44-08-805221",
    # "outputs/2023-11-01/16-24-19-171848",
    "outputs/2023-10-18/11-44-08-874258",
    "outputs/2023-10-18/11-44-08-874396",
    "outputs/2023-10-18/11-44-08-882528",
    "outputs/2023-10-18/11-44-08-884445",
    "outputs/2023-10-18/11-44-08-886024",
    "outputs/2023-10-18/11-44-08-906468",
    "outputs/2023-10-18/11-44-08-911000",
    "outputs/2023-10-18/11-44-08-932201",
    "outputs/2023-10-18/11-44-08-953521",
    # "outputs/2023-11-12/19-20-15-010722",
    "outputs/2023-11-13/12-05-39-054691",
    "outputs/2023-11-14/10-27-23-673379",
]


def read_exp_folders(paths):
    exp_folders = {}
    for path in paths:
        yaml_file = os.path.join(path, ".hydra", "overrides.yaml")
        # parse file
        with open(yaml_file, "r") as stream:
            try:
                data = yaml.safe_load(stream)
                experiment = data[-1]
                experiment = experiment.split("=")[1]
                experiment = experiment.replace("dfa/", "")
                layer = data[4]
                layer = layer.split("=")[1]
                exp_folders[f"{experiment}/{layer}"] = path
                # ckpt = glob.glob(os.path.join(path, "finalexps2", "*/checkpoints/*.ckpt"))[0]
                # # escape equal sign
                # ckpt = ckpt.replace("=", "\\=")
                # overrides = " ".join(data)
                # print(f"export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=2; python train.py train.ckpt=\"{ckpt}\" train.test=true {overrides} > newexps/{experiment}_{layer}.log 2> newexps/{experiment}_{layer}.err &")
            except yaml.YAMLError as exc:
                print(exc)
    return exp_folders


def plot_clusters(
    folder,
    reduction="tsne",
    layer_wise=False,
    factors=None,
    fill_types=None,
    colors=None,
):
    results = get_results(os.path.join(folder, "generations", "200_test.txt"))
    assert len(results) == 500
    os.makedirs(os.path.join(folder, "plots"), exist_ok=True)
    for i in range(2):
        example = results[i]
        t_hidden = len(example["hidden_outputs"][0])
        t_states = len(example["states"])
        n_layers = len(example["hidden_outputs"])

        reducer = TSNE if reduction == "tsne" else PCA

        if layer_wise:
            Xs = [
                reducer(n_components=2).fit_transform(hidden_outputs)
                for hidden_outputs in example["hidden_outputs"]
            ]
            X = np.concatenate(Xs, axis=0)
            X = X.reshape(n_layers, t_hidden, 2)
        else:
            hidden_outputs = np.concatenate(example["hidden_outputs"], axis=0)
            X = reducer(n_components=2).fit_transform(hidden_outputs)
            X = X.reshape(n_layers, t_hidden, 2)

        X = X[:, :t_states, :]

        for t in range(20, t_states, 10):
            a, _ = factors[n_layers]
            axes, fig = plt.subplots(a, a, figsize=(16, 16))
            # assert n_layers == 13
            axes.suptitle(
                f"T={t}, #states: "
                + str(len(example["dfa"].dfa._transition_function))
                + ", len(vocab): "
                + str(len(example["dfa"].dfa.alphabet))
            )

            for layer in range(n_layers - 1, -1, -1):
                X_layer = X[layer, :t]
                labels = np.array(example["states"][:t])
                chars = np.array(list(example["input"][:t]))
                vocab = example["vocab"][1:]

                ax = fig[layer // a, layer % a]
                for label in np.unique(labels):
                    if label == -1 or label == "-1":
                        continue
                    for ci, v in enumerate(example["dfa"].dfa.alphabet):
                        indices = np.where((labels == label) & (chars == v))
                        if len(indices) > 0:
                            ax.scatter(
                                X_layer[indices, 0],
                                X_layer[indices, 1],
                                marker=fill_types[ci],
                                c=colors[label],
                            )
                            # show legend
                # set title
                ax.set_title(f"layer={layer}")
            plt.savefig(
                os.path.join(folder, "plots/", f"e_{i}_{reduction}_out_label_t_{t}.png")
            )
            plt.close()

        video_out_folder = os.path.join(
            folder, f"res_lw_{layer_wise}_videos", f"{reduction}_e_{i}_time"
        )
        video_glob = (
            os.path.join(folder, "plots/")
            + "/"
            + f"e_{i}_{reduction}_out_label_t_*.png"
        )
        prepare_video(video_glob, video_out_folder)
        print(f"done at {video_out_folder}")

    return os.path.join(folder, f"res_lw_{layer_wise}_videos")


if __name__ == "__main__":
    exp_folders = read_exp_folders(PATHS)
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
        9: (3, 3),
        13: (4, 4),
        5: (3, 3),
        3: (2, 2),
        1: (1, 1),
    }

    video_folders = []
    for exp_name, folder in exp_folders.items():
        print(exp_name)
        try:
            output_video_folder = plot_clusters(
                folder,
                reduction="tsne",
                layer_wise=True,
                factors=factors,
                fill_types=fill_types,
                colors=colors,
            )
            video_folders.append(output_video_folder)
        except Exception as e:
            print(e)
            print(f"error, and skipping {exp_name}")

    print("\n".join(video_folders))
