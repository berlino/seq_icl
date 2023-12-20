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

import concurrent.futures
import pickle
import functools


def read_one(fname, probs_only=False, layer=None, key=None):
    with open(fname, "rb") as f:
        data =  pickle.load(f)

    if probs_only:
        for key_other in ["hidden_outputs", "attention_scores", "attention_contexts"]:
            if key_other in data:
                del data[key_other]

    if key is not None:
        for key_other in ["hidden_outputs", "attention_scores", "attention_contexts"]:
            if key_other != key:
                if key_other in data:
                    del data[key_other]

        if layer is not None:
            data[key] = data[key][layer: layer + 1]


    return data


def read_parallel(file_names, probs_only=False, layer=None, key=None):
    reader = functools.partial(read_one, probs_only=probs_only, layer=layer, key=key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(reader, f) for f in file_names]
        return [fut.result() for fut in futures]


def read_hidden_states(folder, probs_only=False, layer=None, key=None):
    files = glob.glob(folder + "/*.pkl")
    ids = [int(os.path.basename(file).replace(".pkl", "")) for file in files]
    # sort files with ids
    files = [file for _, file in sorted(zip(ids, files))]
    content = read_parallel(files, probs_only=probs_only, layer=layer, key=key)
    vocab = content[0]["vocab"]
    assert vocab == content[1]["vocab"]
    return content


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

    assert len(states) == len(input), (len(states), len(input))

    return states


def get_results(file, probs_only=False, layer=None, key=None, subset=None):
    basename = os.path.basename(file)
    basename = (
        basename.replace("_train.txt", "")
        .replace("_test.txt", "")
        .replace("_val.txt", "")
        .replace("_test_batch", "")
        .replace("_val_batch", "")
    )

    fileid = int(basename)

    if subset is not None:
        if subset == "test":
            file = file.replace("val", "test")
        elif subset == "val":
            file = file.replace("test", "val")


    df = pd.read_csv(
        file.replace("_batch", ".txt"),
        sep="\t",
        header=None,
        names=[
            "input",
            "target",
            "pred",
        ],
    )
    pkl_folder = file.replace(".txt", "_batch")

    content = read_hidden_states(pkl_folder, probs_only=probs_only, layer=layer, key=key)
    vocab = content[0]["vocab"]

    batch_size = len(content[0]["probs"])
    assert batch_size == 32


    data = []
    for index, row in df.iterrows():
        pkl_index, batch_index = divmod(index, batch_size)
        if pkl_index >= len(content) or batch_index >= len(content[pkl_index]["probs"]):
            break
        datum = {}
        datum["input"] = row["input"]
        datum["target"] = row["target"]
        datum["pred"] = row["pred"]
        datum["vocab"] = vocab
        datum["dfa"] =  content[pkl_index]["dfas"][batch_index]
        datum["char_labels"] = content[pkl_index]["char_labels"][batch_index]
        datum["probs"] = content[pkl_index]["probs"][batch_index]
        datum["states"] = get_dfa_states(datum["input"], datum["dfa"], in_states=False)

        for key_other in ["hidden_outputs", "attention_scores", "attention_contexts"]:
            if key_other in content[pkl_index]:
                key_values = content[pkl_index][key_other]
                key_values = [states[batch_index] for states in key_values]
                datum[key_other] = key_values

        # if "hidden_outputs" in content[pkl_index]:
        #     hidden_states = content[pkl_index]["hidden_outputs"]
        #     hidden_states = [states[batch_index] for states in hidden_states]
        #     datum["hidden_outputs"] = hidden_states

        # if "attention_scores" in content[pkl_index]:
        #     attentions = content[pkl_index]["attention_scores"]
        #     attentions = [states[batch_index] for states in attentions]
        #     datum["attention_scores"] = attentions

        # if "attention_contexts" in content[pkl_index]:
        #     contexts = content[pkl_index]["attention_contexts"]
        #     contexts = [states[batch_index] for states in contexts]

        #     datum["attention_contexts"] = contexts

        if "hidden_outputs" in datum:
            assert len(datum["input"]) <= datum["hidden_outputs"][0].shape[0], (
                len(datum["input"]),
                datum["hidden_outputs"][0].shape[0],
            )
            assert len(datum["states"]) <= datum["hidden_outputs"][0].shape[0], (
                len(datum["states"]),
                datum["hidden_outputs"][0].shape[0],
            )
        data.append(datum)

    return data


class ProbeModel(nn.Module):
    def __init__(self, nhid, dropout=0.1, ngram=False):
        super(ProbeModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ngram = ngram

        if self.ngram > 1:
            self.embedding = nn.Embedding(20, nhid // 2)
            self.project = nn.Linear(nhid, ngram * nhid // 2)
            self.fc1 = nn.Linear(3 * ngram * nhid // 2, nhid)
        else:
            self.embedding = nn.Embedding(20, nhid)
            self.project = nn.Linear(nhid, nhid)
            self.fc1 = nn.Linear(3 * nhid, nhid)

        self.fc2 = nn.Linear(nhid, 1, bias=False)

    def forward(self, hiddens, chars):
        char_embeds = self.embedding(chars)
        if char_embeds.dim() == 3:
            char_embeds = char_embeds.reshape(char_embeds.shape[0], -1)
        hidden_embeds = self.project(self.dropout(hiddens))
        x = torch.cat((char_embeds, hidden_embeds, char_embeds * hidden_embeds), dim=1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


class StateProbeDataset(Dataset):
    def __init__(self, hiddens, states, chars, vocab, use_ratio=False, ngram=False, binary=False):
        self.hiddens = hiddens
        self.states = states
        self.chars = chars
        self.vocab = vocab
        self.use_ratio = use_ratio
        self.ngram = ngram
        self.binary = binary

        assert len(self.hiddens) == len(self.states)
        assert len(self.hiddens) == len(self.chars)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state_info = self.states[index]
        char_info = ""
        while len(char_info) < 2:
            time_step = np.random.choice(list(range(75, len(state_info))))
            char_info = self.chars[index][: time_step + 1]

        state = state_info[time_step]
        hidden = torch.tensor(self.hiddens[index][time_step])

        if self.ngram == 2:
            char1 = char_info[-1]
            gram2_points = []
            for t in range(len(char_info) - 1):
                if char_info[t] == char1:
                    gram2_points.append(t)
            if self.binary:
                if np.random.rand() > 0.5:
                    if len(gram2_points) > 0:
                        t = np.random.choice(gram2_points)
                        char2 = char_info[t + 1]
                        count = 1
                    else:
                        char2 = np.random.choice(char_info)
                        count = -1
                else:
                    char2s = set([char_info[t + 1] for t in gram2_points])
                    non_char2s = set(char_info).difference(char2s)
                    if len(non_char2s) > 0:
                        char2 = np.random.choice(list(non_char2s))
                        count = -1
                    else:
                        char2 = np.random.choice(list(char2s))
                        count = 1

                tlen = 1
            else:
                if len(gram2_points) > 0:
                    t = np.random.choice(gram2_points)
                    char2 = char_info[t + 1]
                    count = [char_info[point + 1] for point in gram2_points].count(char2)
                else:
                    char2 = np.random.choice(char_info)
                    count = 0
                tlen = max(len(gram2_points), 1)

            char1 = self.vocab.index(char1)
            char2 = self.vocab.index(char2)
            char = [char1, char2]
        elif self.ngram == 1:
            # sample a bigram
            char = char_info[-1]
            count = char_info.count(char)
            char = self.vocab.index(char)
            tlen = len(char_info)
        else:
            # sample an existing bigram
            # t = np.random.choice(list(range(1, len(char_info))))
            char2 = char_info[-1]
            char1 = char_info[-2]
            gram3_points = []
            for t in range(len(char_info) - 2):
                if char_info[t] == char1 and char_info[t + 1] == char2:
                    gram3_points.append(t)

            if self.binary:
                if np.random.rand() > 0.5:
                    if len(gram3_points) > 0:
                        t = np.random.choice(gram3_points)
                        char3 = char_info[t + 2]
                        count = 1
                    else:
                        char3 = np.random.choice(char_info)
                        count = -1
                else:
                    char3s = set([char_info[t + 2] for t in gram3_points])
                    non_char3s = set(char_info).difference(char3s)
                    if len(non_char3s) > 0:
                        char3 = np.random.choice(list(non_char3s))
                        count = -1
                    else:
                        char3 = np.random.choice(list(char3s))
                        count = 1
                tlen = 1
            else:
                if len(gram3_points) > 0:
                    t = np.random.choice(gram3_points)
                    char3 = char_info[t + 2]
                    count = [char_info[point + 2] for point in gram3_points].count(char3)
                else:
                    char3 = np.random.choice(char_info)
                    count = 0
                tlen = max(len(gram3_points), 1)

            char1 = self.vocab.index(char1)
            char2 = self.vocab.index(char2)
            char3 = self.vocab.index(char3)
            char = [char1, char2, char3]

        # hidden[:] = 0
        # hidden[0] = ratio
        return hidden, char, count, tlen

    def collate_fn(self, batch):
        hiddens, chars, counts, totals = zip(*batch)
        hiddens = torch.stack(hiddens, dim=0)
        if hiddens.dim() == 3:
            hiddens = hiddens.reshape(hiddens.shape[0], -1)
        chars = torch.LongTensor(chars)
        counts = torch.tensor(counts).float()
        totals = torch.tensor(totals).float()
        return hiddens, chars, counts, totals


def train(args, hiddens, states, chars, vocab):
    # init Transformer Encoder with causal masking
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # model
    dataset = StateProbeDataset(
        hiddens, states, chars, vocab, use_ratio=args.use_ratio, ngram=args.ngram, binary=args.binary
    )
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

    nhid = hiddens[0][0].reshape(-1).shape[0]
    model = ProbeModel(nhid=nhid, ngram=args.ngram)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.n_epochs, eta_min=args.min_lr
    )
    model.train()

    for e in range(args.n_epochs):
        for hidden, char, count, total_count in train_loader:
            optimizer.zero_grad()
            logits = model(hidden.cuda(), char.cuda())
            if args.use_ratio:
                target = count / total_count
            else:
                target = count

            target = target.cuda()

            if args.ngram != 1 and not args.binary:
                mask = (count != 0).float().cuda()
            loss = F.mse_loss(logits[:, 0], target, reduction="none")
            if args.ngram != 1 and not args.binary:
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # validation
        total = 0.0
        val_loss = 0.0
        model.eval()
        for hidden, char, count, total_count in val_loader:
            logits = model(hidden.cuda(), char.cuda())
            if args.use_ratio:
                target = count / total_count
            else:
                target = count

            target = target.cuda()

            if args.ngram == 1:
                errors = torch.abs((logits[:, 0] - target))  /  (target + 1e-5)
                val_loss += (errors).sum().item()
                total += len(errors)
                metric = "error"
            elif args.binary:
                errors = (logits[:, 0] > 0) == (target > 0)
                val_loss += (errors).sum().item()
                total += len(errors)
                metric = "acc"
            else:
                mask = (count != 0).float().cuda()
                errors = torch.abs((logits[:, 0] - target))  /  (target + 1e-5)
                val_loss += (errors * mask).sum().item()
                total += mask.sum().item()
                metric = "error"

        val_loss /= total

        if args.use_wandb:
            wandb.log({f"val/{metric}": val_loss})
        else:
            print({f"val/{metric}": val_loss})
        model.train()

    if args.use_wandb:
        wandb.log({"val_loss_final": val_loss})
    else:
        print("val_loss_final", val_loss)
    return model, optimizer


def evaluate(args, model, hiddens, states, chars, vocab):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # model
    dataset = StateProbeDataset(
        hiddens, states, chars, vocab, use_ratio=args.use_ratio, ngram=args.ngram, binary=args.binary
    )

    test_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    total = 0.0
    test_loss = 0.0
    model.eval()
    for _ in range(20):
        for hidden, char, count, total_count in test_loader:
            logits = model(hidden.cuda(), char.cuda())
            if args.use_ratio:
                target = count / total_count
            else:
                target = count

            target = target.cuda()

            if args.ngram == 1:
                errors = torch.abs((logits[:, 0] - target))  /  (target + 1e-5)
                test_loss += (errors).sum().item()
                total += len(errors)
                metric = "error"
            elif args.binary:
                errors = (logits[:, 0] > 0) == (target > 0)
                test_loss += (errors).sum().item()
                total += len(errors)
                metric = "acc"
            else:
                mask = (count != 0).float().cuda()
                errors = torch.abs((logits[:, 0] - target))  /  (target + 1e-5)
                test_loss += (errors * mask).sum().item()
                total += mask.sum().item()
                metric = "error"

    test_loss /= total

    if args.use_wandb:
        wandb.log({f"test/final_{metric}": test_loss})
    else:
        print({f"test/final_{metric}": test_loss})

def prepare_data(results):
    assert len(results[0][args.hidden_key]) == 1, len(results[0][args.hidden_key])
    hiddens = [result[args.hidden_key][0] for result in results]
    states = [result["states"] for result in results]
    chars = [list(result["input"]) for result in results]
    vocab = results[0]["vocab"]
    return hiddens, states, chars, vocab

def run(args, training, testing):
    hiddens, states, chars, vocab = prepare_data(training)
    model, optimizer = train(args, hiddens, states, chars, vocab)
    hiddens, states, chars, vocab = prepare_data(testing)
    evaluate(args, model, hiddens, states, chars, vocab)






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="transformer/12")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_ratio", action="store_true")
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--hidden_key", type=str, default="hidden_outputs")
    parser.add_argument("--binary", action="store_true")

    args = parser.parse_args()
    if args.use_wandb:
        import wandb

        wandb.init(project="interpret_dfa_all_probes", config=args)
        wandb.config.update(args)

    # exp_folders = {'transformer/8': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-320622',
    #                'transformer/2': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-041944',
    #                'transformer/4': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-295893',
    #                'transformer/1': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-403698',
    #                'linear_transformer/4': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-52-854931',
    #                'retnet/4': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-21-36-646480',
    #                'rwkv/2': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-21-36-588119',
    #                'h3/2': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-27-29-253904',
    #                'hyena/2': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-21-36-614857',
    #                'lstm/1': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-00-28-036885',
    #                'transformer/12': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-222033',
    #                'linear_transformer/8': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-201063',
    #                'lstm/3': '/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-28/11-12-43-061481'}

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
    }

    training_data = get_results(exp_folders_40000[args.exp], subset="val", layer=args.layer, key=args.hidden_key)
    testing_data = get_results(exp_folders_40000[args.exp], subset="test", layer=args.layer, key=args.hidden_key)

    run(args, training_data, testing_data)


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
