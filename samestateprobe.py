import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from analyze import eval_dfa
from pythomata import SimpleDFA
from src.dataloaders.dfa import DFA

def read_hidden_states(folder):
    files = glob.glob(folder + "/*.pkl")
    ids = [int(os.path.basename(file).replace(".pkl", "")) for file in files]
    # sort files with ids
    files = [file for _, file in sorted(zip(ids, files))]
    hidden_states = []
    dfas = []
    char_labels = []
    probs = []
    vocab = None
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)
            hidden_states.append(data["hidden_outputs"])
            dfas += data["dfas"]
            char_labels += data["char_labels"]
            probs.append(data["probs"])
            if vocab is None:
                vocab = data["vocab"]

    probs = np.concatenate(probs, axis=0)

    data = []
    for layer in range(len(hidden_states[0])):
        # concat all hidden states
        layer_states = [state[layer] for state in hidden_states]
        layer_states = np.concatenate(layer_states, axis=0)
        data.append(layer_states)
    return data, dfas, char_labels, probs, vocab


def get_dfa_states(input, dfa, in_states=False):
    examples = input.split("|")
    states = []
    for example in examples:
        state = dfa.trace(" ".join(list(example)))
        if in_states:
            states.extend(state[:-1])
        else:
            states.extend(state[1:])
        states.append(-1)
    states = states[:-1]
    assert len(states) == len(input)
    return states

def get_results(file):
    basename = os.path.basename(file)
    basename = basename.replace("_train.txt", "")
    basename = basename.replace("_test.txt", "")
    basename = basename.replace("_val.txt", "")
    fileid = int(basename)
    df = pd.read_csv(
        file,
        sep="\t",
        header=None,
        names=[
            "input",
            "target",
            "pred",
            "dfa",
            "diff_n_gram",
            "diff_dfa",
            "diff_dfa_ngram",
        ],
    )
    pkl_folder = file.replace(".txt", "_batch")
    hidden_states, dfas, char_labels, probs, vocab = read_hidden_states(pkl_folder)
    df["dfa"] = dfas
    df["char_labels"] = char_labels
    data = []
    for index, row in df.iterrows():
        datum = {}
        datum["input"] = row["input"]
        datum["target"] = row["target"]
        datum["pred"] = row["pred"]
        datum["dfa"] = row["dfa"]
        datum["char_labels"] = row["char_labels"]
        datum["probs"] = probs[index]
        datum["vocab"] = vocab
        datum["states"] = get_dfa_states(datum["input"], datum["dfa"], in_states=False)
        if hidden_states is not None:
            datum["hidden_outputs"] = [states[index] for states in hidden_states]
        assert len(datum["input"]) <= datum["hidden_outputs"][0].shape[0], (len(datum["input"]), datum["hidden_outputs"][0].shape[0])
        assert len(datum["states"]) <= datum["hidden_outputs"][0].shape[0], (len(datum["states"]), datum["hidden_outputs"][0].shape[0])
        data.append(datum)

    return data


class ProbeModel(nn.Module):
    def __init__(self, nhid, dropout=0.1, bigram=False):
        super(ProbeModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.project1 = nn.Linear(nhid, nhid)

        self.fc1 = nn.Linear(3 * nhid, nhid)
        self.fc2 = nn.Linear(nhid, 2, bias=False)

    def forward(self, hiddens1, hiddens2):
        hiddens1 = self.project1(self.dropout(hiddens1))
        hiddens2 = self.project1(self.dropout(hiddens2))
        x = torch.cat((hiddens1, hiddens2, hiddens1 * hiddens2), dim=1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


class StateProbeDataset(Dataset):
    def __init__(self, hiddens, states, chars, vocab, use_ratio=False, bigram=False):
        self.hiddens = hiddens
        self.states = states
        self.chars = chars
        self.vocab = vocab
        self.use_ratio = use_ratio
        self.bigram = bigram
        assert len(self.hiddens) == len(self.states)
        assert len(self.hiddens) == len(self.chars)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state_info = self.states[index]

        if np.random.rand() < 0.5:
            time_step1 = 0
            time_step2 = 0
            time_step1, time_step2 = np.random.choice(list(range(1, len(state_info))), 2)
            state1 = state_info[time_step1]
            state2 = state_info[time_step2]
            label = state1 == state2
        else:
            # sample same states
            state = np.random.choice(list(set(state_info) - {-1}))
            # find timesteps that matches the sate
            time_steps = np.where(state_info == state)[0]
            # sample two random time steps
            time_step1, time_step2 = np.random.choice(time_steps, size=2, replace=True)
            state1 = state_info[time_step1]
            state2 = state_info[time_step2]
            label = state1 == state2

        hidden1 = self.hiddens[index][time_step1]
        hidden2 = self.hiddens[index][time_step2]

        return torch.tensor(hidden1), torch.tensor(hidden2), int(label)

    def collate_fn(self, batch):
        hiddens1, hiddens2, label = zip(*batch)
        hiddens1 = torch.stack(hiddens1, dim=0)
        hiddens2 = torch.stack(hiddens2, dim=0)
        labels = torch.tensor(label, dtype=torch.long)
        return hiddens1, hiddens2, labels

def train(args, hiddens, states, chars, vocab):
    # init Transformer Encoder with causal masking
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # model
    dataset = StateProbeDataset(hiddens, states, chars, vocab, use_ratio=args.use_ratio, bigram=args.bigram)
    # split
    train_size = int(0.95 * len(dataset))
    train = torch.utils.data.Subset(dataset, list(range(train_size)))
    val = torch.utils.data.Subset(dataset, list(range(train_size, len(dataset))))
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )

    model = ProbeModel(nhid=128, bigram=args.bigram)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.n_epochs, eta_min=args.min_lr
    )
    model.train()

    for e in range(args.n_epochs):
        for hiddens1, hiddens2, labels in train_loader:
            optimizer.zero_grad()
            logits = model(hiddens1.cuda(), hiddens2.cuda())
            loss = F.cross_entropy(logits, labels.cuda())
            loss.backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        # print("learning rate:", scheduler.get_last_lr()[0])

        # validation
        total = 0.0
        val_corrects = 0.0
        model.eval()
        for hiddens1, hiddens2, labels in val_loader:
            logits = model(hiddens1.cuda(), hiddens2.cuda())
            preds = torch.argmax(logits, dim=1)
            corrects = preds == labels.cuda()
            total += hiddens1.shape[0]
            val_corrects += corrects.sum().item()
        val_corrects /= total
        if args.use_wandb:
            wandb.log({"val_acc": val_corrects})
        else:
            print("val acc:", val_corrects)
        model.train()

    wandb.log({"val_acc_final": val_corrects})
    return model, optimizer

def run(args, results):
    hiddens = [result["hidden_outputs"][args.layer] for result in results]
    states = [result["states"] for result in results]
    chars = [list(result["input"]) for result in results]
    vocab = results[0]["vocab"]
    return train(args, hiddens, states, chars, vocab)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="transformers/12")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_ratio", action="store_true")
    parser.add_argument("--bigram", action="store_true")


    args = parser.parse_args()
    if args.use_wandb:
        import wandb
        wandb.init(project="dfa_ss_probe", config=args)
        wandb.config.update(args)

    exp_folders = {'transformer/8': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-320622',
                   'transformer/2': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-041944',
                   'transformer/4': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-295893',
                   'transformer/1': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-403698',
                   'linear_transformer/4': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-52-854931',
                   'retnet/4': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-21-36-646480',
                   'rwkv/2': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-21-36-588119',
                   'h3/2': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-27-29-253904',
                   'hyena/2': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-21-36-614857',
                   'lstm/1': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-00-28-036885',
                   'transformer/12': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-222033',
                   'linear_transformer/8': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-201063'}

    results = get_results(exp_folders[args.exp] + "/generations/200_val.txt")
    model, optimizer = run(args, results)


# def minimize(dfa):
#     transitions = {i: v for i, v in enumerate(dfa.transitions)}
#     sdfa = SimpleDFA(
#         states = set(list(range(dfa.num_nodes))),
#         alphabet = set(dfa.alphabet),
#         initial_state = 0,
#         accepting_states = set(list(range(dfa.num_nodes))),
#         transition_function = transitions,
#     )
#     sdfa = sdfa.minimize().trim()
#     # convert back to our data structure
#     states = [state for state in sdfa.states if state != sdfa.initial_state]
#     states = [sdfa.initial_state] + states
#     transitions = []
#     for index, state in enumerate(states):
#         new_transitions = {}
#         old_transitions = sdfa._transition_function[state]
#         for symbol, next_state in old_transitions.items():
#             new_transitions[symbol] = states.index(next_state)
#         transitions.append(new_transitions)
#     transitions = tuple(transitions)
#     alphabet = tuple(sorted(list(sdfa.alphabet)))
#     dfa = DFA(
#         num_nodes=len(states),
#         alphabet=alphabet,
#         transitions=transitions,
#         rng=np.random.RandomState(0),
#     )
#     return dfa

# class SameStateProbeDataset(Dataset):
#     def __init__(self, hiddens, states):
#         self.hiddens = hiddens
#         self.states = states
#         assert len(self.hiddens) == len(self.states)

#     def __len__(self):
#         return len(self.states)

#     def __getitem__(self, index):
#         if np.random.rand() < 0.5:
#             state = np.random.choice(list(set(self.states[index]) - {-1}))
#             # sample random two indices with the same state
#             indices = np.where(self.states[index] == state)[0]
#             i, j = np.random.choice(indices, size=2, replace=True)
#             hidden1 = torch.tensor(self.hiddens[index][i])
#             hidden2 = torch.tensor(self.hiddens[index][j])
#             label = 1
#         else:
#             # sample two random indices
#             i, j = np.random.choice(len(self.states[index]), size=2, replace=False)
#             hidden1 = torch.tensor(self.hiddens[index][i])
#             hidden2 = torch.tensor(self.hiddens[index][j])
#             label = int(self.states[index][i] == self.states[index][j])

#         return hidden1, hidden2, label

#     def collate_fn(self, batch):
#         hidden1s, hidden2s, labels = zip(*batch)
#         hidden1s = torch.stack(hidden1s, dim=0)
#         hidden2s = torch.stack(hidden2s, dim=0)
#         labels = torch.LongTensor(labels)
#         return hidden1s, hidden2s, labels