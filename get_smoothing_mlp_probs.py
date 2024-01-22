import numpy as np
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import pickle
from probe import get_results
from analyze import get_dfa_probs as calculate_dfa_probs

N_VOCAB = 20
N_HIDDEN = 1024


class Vocab:
    def __init__(self, vocab: list):
        self.vocab = vocab
        # inverse vocab
        self.inv_vocab = {v: k for k, v in enumerate(vocab)}

    def get_vocab(self, id):
        return self.vocab[id]

    def get_id(self, char):
        return self.inv_vocab[char]

    def __len__(self):
        return len(self.vocab)


class Smoother(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # nn.LayerNorm(3 * N_VOCAB),
            nn.Linear(3 * N_VOCAB, N_HIDDEN),
            nn.GELU(),
            nn.Linear(N_HIDDEN, N_VOCAB),
            # nn.LogSoftmax()
        )
        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
        self.tf_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.tf_l1_loss = nn.L1Loss()

    def _accumulate_counts(self, datum):
        chars = [datum["vocab"].index(c) for c in datum["input"]]
        chars_tgt = [datum["vocab"].index(c) for c in datum["target"]]
        # mask "|" chars with -100
        imid = datum["vocab"].index("|")
        chars_tgt = [-100 if c == imid else c for c in chars_tgt]
        n_chars = len(chars)
        unigrams = torch.zeros(n_chars, N_VOCAB)
        bigrams = torch.zeros(n_chars, N_VOCAB)
        trigrams = torch.zeros(n_chars, N_VOCAB)
        unigrams_cum = torch.zeros(N_VOCAB)
        bigrams_cum = torch.zeros(N_VOCAB, N_VOCAB)
        trigrams_cum = torch.zeros(N_VOCAB, N_VOCAB, N_VOCAB)

        hist = [datum["vocab"].index(c) for c in "|||"]
        for i in range(n_chars):
            curr_char = chars[i]
            hist.append(curr_char)
            hist.pop(0)
            unigrams_cum[hist[-1]] += 1
            bigrams_cum[hist[-2], hist[-1]] += 1
            trigrams_cum[hist[-3], hist[-2], hist[-1]] += 1
            unigrams[i, :] = unigrams_cum  # / (unigrams_cum.sum() + 1e-4)
            bigrams[i, :] = bigrams_cum[
                hist[-1], :
            ]  # / (bigrams_cum[hist[-1], :].sum() + 1e-4)
            trigrams[i, :] = trigrams_cum[
                hist[-2], hist[-1], :
            ]  # / (trigrams_cum[hist[-2], hist[-1], :].sum() + 1e-4)

        if args.use_ratio:
            unigrams = F.normalize(unigrams, dim=1, p=1)
            bigrams = F.normalize(bigrams, dim=1, p=1)
            trigrams = F.normalize(trigrams, dim=1, p=1)
        elif args.use_binary:
            unigrams = (unigrams > 0).float()
            bigrams = (bigrams > 0).float()
            trigrams = (trigrams > 0).float()
        else:
            unigrams = unigrams / 100
            bigrams = bigrams / 100
            trigrams = trigrams / 100

        counts = torch.cat([unigrams, bigrams, trigrams], dim=1)
        # counts = torch.cat([unigrams, bigrams], dim=1)
        # counts_norm = counts / 100
        return counts, torch.tensor(chars_tgt)

    def forward(self, batch):
        batch_counts = []
        batch_targets = []
        for seq in batch:
            counts, targets = self._accumulate_counts(seq)
            batch_counts.append(counts)
            batch_targets.append(targets)
        # pad
        counts = torch.nn.utils.rnn.pad_sequence(batch_counts, batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(
            batch_targets, batch_first=True, padding_value=-100
        )

        counts = counts.cuda()
        targets = targets.cuda()
        preds = self.layers(counts)
        # target_probs = seq["probs"][:preds.size(0), :]
        # target_probs = torch.log(target_probs + 1e-7)
        # tf_preds = target_probs.cuda()
        # flatten for loss
        pred_shape = preds.shape
        preds = preds.view(-1, preds.size(-1))
        targets = targets.view(-1)
        loss = self.loss(preds, targets)
        # loss = self.tf_loss(preds.log_softmax(dim=1), tf_preds)
        # loss = self.tf_l1_loss(preds.softmax(dim=1), tf_preds.softmax(dim=1))
        assert loss >= 0

        preds = preds.reshape(pred_shape)

        return loss, preds, len(targets)


def get_dfa_probs(results):
    vocab = Vocab(results[0]["vocab"])
    dfa_probs = []
    for b in range(len(results)):
        input = results[b]["input"]
        target = [vocab.get_id(t) for t in results[b]["target"]]
        probs = calculate_dfa_probs(input, results[b]["dfa"], vocab=vocab)
        dfa_probs.append(probs)
    return dfa_probs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="transformer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_ratio", action="store_true")
    parser.add_argument("--use_binary", action="store_true")
    parser.add_argument("--num_examples", type=int, default=40000)

    args = parser.parse_args()
    if args.use_wandb:
        import wandb
        wandb.init(project=f"smoothing_{args.num_examples}", config=args)
        wandb.config.update(args)

    exp_folders_1000 = {
        "transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_1000/transformer/generations/46_test.txt",
    }

    exp_folders_2500 = {
        "transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_2500/transformer/generations/194_test.txt",
    }

    exp_folders_5000 = {
        "transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_5000/transformer/generations/194_test.txt",
    }

    exp_folders_10000 = {
        "transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_10000/transformer_8/generations/161_test.txt",
    }

    exp_folders_20000 = {
        "transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_20000/transformer_8/generations/192_test.txt",
    }

    exp_folders_40000 = {
        "transformer": "/raid/lingo/akyurek/git/iclmodels/experiments/hiddens_40000/transformer_12/generations/184_test.txt",
    }

    if args.num_examples == 2500:
        exp_folders = exp_folders_2500
    elif args.num_examples == 10000:
        exp_folders = exp_folders_10000
    elif args.num_examples == 1000:
        exp_folders = exp_folders_1000
    elif args.num_examples == 5000:
        exp_folders = exp_folders_5000
    elif args.num_examples == 20000:
        exp_folders = exp_folders_20000
    elif args.num_examples == 40000:
        exp_folders = exp_folders_40000
    else:
        raise ValueError("invalid num_examples")

    train_data = get_results(exp_folders[args.exp], subset="val", probs_only=True)
    tf_data = get_results(exp_folders[args.exp], subset="test", probs_only=True)

    # replace probs with dfa probs
    train_dfa_probs = get_dfa_probs(train_data)
    tf_dfa_probs = get_dfa_probs(tf_data)
    for i in range(len(train_data)):
        train_data[i]["probs"] = train_dfa_probs[i]
    for i in range(len(tf_data)):
        tf_data[i]["probs"] = tf_dfa_probs[i]

    batch_size = args.batch_size
    smoother = Smoother().cuda()
    opt = optim.Adam(smoother.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=args.min_lr
    )
    for i_epoch in range(args.n_epochs):
        train_loss = 0.0
        train_total = 0.0
        indices = np.random.permutation(len(train_data))
        for i in tqdm(range(0, len(indices), batch_size)):
            # batch_loss = 0
            # for j in indices[i:i+batch_size]:
            #     datum = train_data[j]
            #     loss, _ = smoother(datum)
            #     batch_loss += loss
            #     train_loss += loss.item()
            batch = [train_data[j] for j in indices[i : i + batch_size]]
            batch_loss, preds, length = smoother(batch)
            opt.zero_grad()
            train_loss += batch_loss.item()
            train_total += length
            batch_loss = batch_loss / length
            batch_loss.backward()
            opt.step()
        train_loss /= train_total

        scheduler.step(train_loss)

        with torch.no_grad():
            test_loss = 0.0
            test_total = 0.0
            for i in tqdm(range(0, len(tf_data), batch_size)):
                batch = tf_data[i : i + batch_size]
                loss, preds, length = smoother(batch)
                test_loss += loss.item()
                test_total += length

        test_loss /= test_total

        print(test_loss)
        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        else:
            print(f"train_loss: {train_loss}, test_loss: {test_loss}")



    all_preds = []
    with torch.no_grad():
        test_loss = 0.0
        test_total = 0.0
        for i in tqdm(range(0, len(tf_data), batch_size)):
            batch = tf_data[i : i + batch_size]
            loss, preds, length = smoother(batch)
            preds = torch.softmax(preds, dim=-1).detach().cpu().numpy()
            for b in range(preds.shape[0]):
                all_preds.append(preds[b, :batch[b]["probs"].shape[0], :])
            test_loss += loss.item()
            test_total += length

    test_loss /= test_total

    if args.use_wandb:
        wandb.log({"final/test_loss": test_loss})
    else:
        print(f"final/test_loss: {test_loss}")

    for pred, data in zip(all_preds, tf_data):
        # trim
        # pred = pred[: len(data["probs"])]
        # take softmax
        # pred = torch.softmax(pred, dim=-1)
        data["probs"] = pred


    suffix = ""
    if args.use_ratio:
        suffix += "_r"
    if args.use_binary:
        suffix += "_b"

    # save as pickle
    save_dir = os.path.join(
        "/raid/lingo/akyurek/git/iclmodels",
        "experiments",
        f"hiddens_{args.num_examples}",
        "smoothing" + suffix,
    )

    #makedirs
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, f"probs.pkl"), "wb") as f:
        pickle.dump(tf_data, f)
