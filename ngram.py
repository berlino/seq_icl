from typing import Mapping, List, Set, Tuple
import numpy as np
from collections import Counter
import torch

def update_ngram_probs_(text: str, counters: Mapping[int, Counter]):
    # give a string, update the ngram counts of characters
    for n in counters.keys():
        counter = counters[n]
        for i in range(len(text)):
            word = text[i : i + n]
            counter[word] += 1


def normalize(counter: Counter, addone: bool = False) -> Mapping[str, float]:
    # normalize a counter to get probs, optionally does add-one smoothing
    probs = counter.copy()
    for word, count in counter.items():
        probs[word] = (count + addone) / (counter.total() + addone * len(counter))
    return probs

def train_everygram(
    N: int, texts: List[str]
) -> Tuple[Mapping[int, Mapping[str, float]], List[str]]:
    # obtain counters for n-grams up to N, and unigram vocabulary
    counters = {n: Counter() for n in range(1, N + 1)}
    for text in texts:
        # adds padding
        text = "_" * (N - 1) + text
        update_ngram_probs_(text, counters)
    vocab = sorted(set(list("".join(texts))))
    return counters, vocab


def get_conditional_prob(
    prefix: str,
    char: str,
    counts_n: Counter,
    counts_n_1: Counter,
    vocab: List[str],
    addone: bool = True,
) -> float:
    # get conditional prob of char given prefix using n and n-1 counts
    # prefix + char should be in counts_n
    assert prefix + char in counts_n
    if counts_n_1 is not None:
        assert prefix in counts_n_1
        return (counts_n[prefix + char] + addone) / (
            counts_n_1[prefix] + (addone * len(vocab))
        )
    else:
        # when n=1
        return normalize(counts_n, addone=addone)[prefix + char]


def get_next_char_prob(
    prefix: str,
    chars: List[str],
    n: int,
    counts: Mapping[int, Mapping[str, float]],
    vocab: List[str],
    backoff: bool = True,
    addone: bool = False,
):
    next_probs = []
    for char in chars:
        query = prefix + char
        if query in counts[n]:
            # if the n-gram exists, use it
            prob = get_conditional_prob(
                prefix, char, counts[n], counts.get(n - 1, None), vocab, addone=addone
            )
        elif backoff:
            # if the n-gram doesn't exist, backoff to lower order n-grams
            # we need to calculate remaining probability mass of the n-gram
            sum_prob = 0.0
            non_exist_chars = []
            for _char in set(vocab):
                other_query = prefix + _char
                if other_query in counts[n]:
                    sum_prob += get_conditional_prob(
                        prefix,
                        _char,
                        counts[n],
                        counts.get(n - 1, None),
                        vocab,
                        addone=addone,
                    )
                else:
                    non_exist_chars.append(_char)
            beta = 1.0 - sum_prob
            assert beta > 0.0

            non_exist_probs = get_next_char_prob(
                prefix[1:],
                non_exist_chars,
                n - 1,
                counts,
                vocab,
                backoff=backoff,
                addone=addone,
            )
            alpha = beta / sum(non_exist_probs)
            char_index = non_exist_chars.index(char)
            prob = alpha * non_exist_probs[char_index]
        else:
            prob = 0.0
        next_probs.append(prob)
    return next_probs


def predict_with_n_gram_back_off(inputs: str, N: int = 3, global_vocab=None) -> str:
    # inputs is in the following form  "absadf|adsfab|...."
    # N is the max n_gram order
    predictions = []
    running_probs = []
    for t in range(1, len(inputs)+1):
        texts = inputs[:t].split("|")
        # get counts and vocab
        counts, vocab = train_everygram(N, texts)
        # get the last N-1 chars
        prefix = texts[-1][-(N - 1) :]
        prefix = "_" * (N - 1 - len(prefix)) + prefix
        # get next char probs
        next_probs = get_next_char_prob(prefix, vocab, N, counts, vocab)
        # distribute probs to global vocab
        next_global_probs = np.zeros(len(global_vocab))
        for char, prob in zip(vocab, next_probs):
            next_global_probs[global_vocab.get_id(char)] = prob
        running_probs.append(next_global_probs)
    return np.array(running_probs)
        # greedy decoding
    #     assert len(next_probs) == len(vocab)
    #     next_char_index = np.argmax(next_probs)
    #     running_probs.append((next_probs, vocab))
    #     if len(inputs) > t and inputs[t] == "|":
    #         predictions.append("|")
    #     else:
    #         predictions.append(vocab[next_char_index])
    # return "".join(predictions), running_probs

def l1_distance(p, q):
    # l1
    return np.sum(np.abs(p-q))


def prob_distance(model_probs, n_gram_probs, inputs):
    diff = 0.0
    total = 0.0
    model_vocab = model_probs.vocab
    model_probs = model_probs.probs
    for t in range(1, len(inputs)-1):
        if inputs[t] != "|":
            prob = torch.softmax(torch.tensor(model_probs[t-1]), dim=-1).numpy() + 0.0
            n_gram_prob, current_vocab = n_gram_probs[t-1]
            n_gram_full_prob = prob.copy()
            for i, char in enumerate(model_vocab):
                if char not in current_vocab:
                    n_gram_full_prob[i] = 0.0
                else:
                    n_gram_full_prob[i] = n_gram_prob[current_vocab.index(char)]
            n_gram_full_prob = n_gram_full_prob / np.sum(n_gram_full_prob)
            diff += l1_distance(n_gram_full_prob, prob)
            total += 1

    return diff / total

def prob_distance_dfa(model_probs, dfa_probs, dfa_alphabet, inputs):
    diff = 0.0
    total = 0.0
    model_vocab = model_probs.vocab
    model_probs = model_probs.probs
    #assert len(dfa_probs) == len(inputs) - 1, f"{len(dfa_probs)} != {len(inputs) - 1}"
    for t in range(1, len(inputs)-2):
        if inputs[t] != "|":
            prob = torch.softmax(torch.tensor(model_probs[t-1]), dim=-1).numpy() + 0.0
            dfa_full_prob = prob.copy()
            for i, char in enumerate(model_vocab):
                if char not in dfa_alphabet:
                    dfa_full_prob[i] = 0.0
                else:
                    dfa_full_prob[i] = dfa_probs[t][dfa_alphabet.index(char)]
            assert np.sum(dfa_full_prob) > 0.0
            diff += l1_distance(dfa_full_prob, prob)
            total += 1

    return diff / total

def prob_distance_dfa_ngram(n_gram_probs, dfa_probs, dfa_alphabet, inputs):
    diff = 0.0
    total = 0.0
    #assert len(dfa_probs) == len(inputs) - 1, f"{len(dfa_probs)} != {len(inputs) - 1}"
    for t in range(1, len(inputs)-2):
        if inputs[t] != "|":
            probs = dfa_probs[t] / np.sum(dfa_probs[t])
            n_gram_prob, current_vocab = n_gram_probs[t-1]
            n_gram_full_prob = probs.copy()
            for i, char in enumerate(dfa_alphabet):
                if char not in current_vocab:
                    n_gram_full_prob[i] = 0.0
                else:
                    n_gram_full_prob[i] = n_gram_prob[current_vocab.index(char)]

            n_gram_full_prob = n_gram_full_prob / np.sum(n_gram_full_prob)
            diff += l1_distance(probs, n_gram_full_prob)
            total += 1

    return diff / total



if __name__ == "__main__":
    from analyze import get_results, get_transition_info, get_dfa_probs
    from tqdm import tqdm

    # exp_folder_tf = "outputs/2023-09-06/02-42-23-025200/generations" # TF
    exp_folder_tf = "outputs/2023-09-24/23-11-21-453058/generations"
    # exp_folder_lstm = "outputs/2023-09-07/02-22-14-205796/generations" # LSTM
    exp_folder_lstm = "outputs/2023-09-19/11-05-57-447748/generations"
    exp_folder_hyena = "outputs/2023-09-20/03-41-51-035199/generations"

    tf_results, tf_files = get_results(exp_folder_tf)
    lstm_results, lstm_files = get_results(exp_folder_lstm)

    corrects = 0.0
    total = 0.0
    total_chars = 0.0
    sum_chars = 0.0
    total_diff = 0.0
    total_dfa_diff = 0.0

    for index, example in tqdm(tf_results[-2].iterrows(), disable=True):
        model_pred = example["pred"]
        model_probs = example["probs"]
        dfa_probs, dfa_vocab = get_dfa_probs(example["input"], example["dfa"])
        example = example[["input", "target", "pred", "dfa"]].copy()
        example["pred"], n_gram_probs = predict_with_n_gram_back_off(example["input"], N=3)
        # print("======ngram pred======")
        # print(example["pred"])
        # print("======model pred======")
        # print(model_pred)
        # print("======LSTM pred======")
        # print(lstm_results[-2].iloc[index]["pred"])
        # print("======Hyena pred======")
        # print(lstm_results[-2].iloc[index]["pred"])

        diff_n_gram = prob_distance(model_probs, n_gram_probs, example["input"])
        diff_dfa = prob_distance_dfa(model_probs, dfa_probs, dfa_vocab, example["input"])
        diff_dfa_n_gram = prob_distance_dfa_ngram(n_gram_probs, dfa_probs, dfa_vocab, example["input"])

        total_diff += diff_n_gram
        total_dfa_diff += diff_dfa
        # ngram_pred = example["pred"]
        # char_sim = [pred == target for pred, target in zip(tf_pred, ngram_pred)]
        # sum_chars += sum(char_sim)
        # total_chars += len(char_sim)
        transitions, correct, length = get_transition_info(example)
        corrects += correct
        total += length
        print("running n gram acc: ", corrects / total)
        # print("running diff (ngram, model)", total_diff / total)
        # print("running diff (dfa, model)", total_dfa_diff / total)

