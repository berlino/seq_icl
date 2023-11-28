from typing import Mapping, List, Set, Tuple
import numpy as np
from collections import Counter
import torch


def update_ngram_probs_(text: str, counters: Mapping[int, Counter]):
    # give a string, update the ngram counts of characters
    for n in counters.keys():
        counter = counters[n]
        for i in range(len(text)):
            if i + n <= len(text):
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
        text = "_" * (N - 1) + text # + "_" * (N - 1)
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
        prob = (counts_n[prefix + char] + addone) / (
            counts_n_1[prefix] + (addone * len(vocab))
        )
    else:
        # when n=1
        prob = normalize(counts_n, addone=addone)[prefix + char]

    return prob


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
    ns = []
    for char in chars:
        query = prefix + char
        if query in counts[n]:
            # if the n-gram exists, use it
            prob = get_conditional_prob(
                prefix, char, counts[n], counts.get(n - 1, None), vocab, addone=addone
            )
            char_by_n = n
        elif backoff:
            # if the n-gram doesn't exist, backoff to lower order n-grams
            # we need to calculate remaining probability mass of the n-gram
            sum_prob = 0.0
            non_exist_chars = []
            for _char in set(vocab): #.union({"_"}):
                other_query = prefix + _char
                if other_query in counts[n]:
                    sum_prob += get_conditional_prob(
                        prefix,
                        _char,
                        counts[n],
                        counts.get(n - 1, None),
                        vocab, # + ["_"],
                        addone=addone,
                    )
                else:
                    non_exist_chars.append(_char)
            beta = 1.0 - sum_prob
            # print(beta)
            # if beta != 1.0:
            #     try:
            #         assert np.abs(1 - (counts.get(n - 1, None)[prefix] - 1) / counts.get(n - 1, None)[prefix] - beta) < 1e-5
            #     except:
            #         breakpoint()
            assert beta > 0.0

            non_exist_probs, non_exist_ns = get_next_char_prob(
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
            char_by_n = non_exist_ns[char_index]
        else:
            prob = 0.0
            char_by_n = n
        next_probs.append(prob)
        ns.append(char_by_n)
    return next_probs, ns


def predict_with_n_gram_back_off(
    inputs: str,
    N: int = 3,
    global_vocab=None,
    backoff: bool = True,
    addone: bool = False,
    uniform: bool = False,
) -> str:
    # inputs is in the following form  "absadf|adsfab|...."
    # N is the max n_gram order
    predictions = []
    running_probs = []
    for t in range(1, len(inputs) + 1):
        texts = inputs[:t].split("|")
        # get counts and vocab
        counts, vocab = train_everygram(N, texts)
        # get the last N-1 chars
        prefix = texts[-1][-(N - 1) :]
        prefix = "_" * (N - 1 - len(prefix)) + prefix
        # get next char probs
        next_probs, next_prob_ns = get_next_char_prob(
            prefix, vocab, N, counts, vocab, backoff=backoff, addone=addone
        )
        # distribute probs to global vocab
        next_probs = np.array(next_probs)
        next_prob_ns = np.array(next_prob_ns)
        # normalize
        # print(np.abs(1-next_probs.sum()))

        if uniform:
            # for each n we want to uniformly spread the distribution
            unique_ns = set(next_prob_ns.tolist()) - {-1}
            for n in unique_ns:
                n_indices = np.where((next_prob_ns == n) & (next_probs != 0))[0]
                if len(n_indices) > 0:
                    n_probs = next_probs[n_indices]
                    n_probs = np.sum(n_probs) / len(n_indices)
                    next_probs[n_indices] = n_probs

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
    return np.sum(np.abs(p - q))


def prob_distance(model_probs, n_gram_probs, inputs):
    diff = 0.0
    total = 0.0
    model_vocab = model_probs.vocab
    model_probs = model_probs.probs
    for t in range(1, len(inputs) - 1):
        if inputs[t] != "|":
            prob = torch.softmax(torch.tensor(model_probs[t - 1]), dim=-1).numpy() + 0.0
            n_gram_prob, current_vocab = n_gram_probs[t - 1]
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
    # assert len(dfa_probs) == len(inputs) - 1, f"{len(dfa_probs)} != {len(inputs) - 1}"
    for t in range(1, len(inputs) - 2):
        if inputs[t] != "|":
            prob = torch.softmax(torch.tensor(model_probs[t - 1]), dim=-1).numpy() + 0.0
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
    # assert len(dfa_probs) == len(inputs) - 1, f"{len(dfa_probs)} != {len(inputs) - 1}"
    for t in range(1, len(inputs) - 2):
        if inputs[t] != "|":
            probs = dfa_probs[t] / np.sum(dfa_probs[t])
            n_gram_prob, current_vocab = n_gram_probs[t - 1]
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
    from probe import get_results
    from analyze import get_dfa_probs as calculate_dfa_probs
    def get_dfa_probs(results):
        vocab = Vocab(results[0]["vocab"])
        dfa_probs = []
        for b in range(len(results)):
            input = results[b]["input"]
            target = [vocab.get_id(t) for t in results[b]["target"]]
            probs = calculate_dfa_probs(input, results[b]["dfa"], vocab=vocab)
            dfa_probs.append(probs)
        return dfa_probs

    import os

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

    def get_ngram_probs(results, ngram=3, uniform=False, backoff=False, addone=False):
        vocab = Vocab(results[0]["vocab"])
        n_gram_probs = []
        for b in range(len(results)):
            input = results[b]["input"]
            target = [vocab.get_id(t) for t in results[b]["target"]]
            probs = predict_with_n_gram_back_off(
                input,
                N=ngram,
                global_vocab=vocab,
                uniform=uniform,
                backoff=backoff,
                addone=addone,
            )
            n_gram_probs.append(probs)
        return n_gram_probs



    def get_greedy_dfa_accuracy(probs, dfa_probs):
        total = 0.0
        correct = 0.0
        for p1, pdfa in zip(probs, dfa_probs):
            indices = p1.argmax(axis=-1)[: len(pdfa)]
            correct += (pdfa[np.arange(len(pdfa)), indices] > 0).sum()
            total += len(pdfa)
        return correct / total

    EPS = 1e-7

    def get_cross_entropy(probs, dfa_probs):
        total = 0.0
        cross_entropy = 0.0
        for p1, pdfa in zip(probs, dfa_probs):
            # calculate the soft cross-entropy between p1 and pdfa
            log_p1 = np.log(p1[: len(pdfa)] + EPS)
            log_pdfa = np.log(pdfa + EPS)
            cross_entropy += -((log_p1 - log_pdfa) * pdfa).sum()
            total += len(pdfa)
        return cross_entropy / total

    exp_folders = {
        "transformer/8": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-320622"
        ),
        "transformer/2": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-041944"
        ),
        "transformer/4": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-295893"
        ),
        "transformer/1": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-403698"
        ),
        "linear_transformer/4": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-52-854931"
        ),
        "retnet/4": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-21-36-646480"
        ),
        "rwkv/2": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-21-36-588119"
        ),
        "h3/2": "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-27-29-253904",
        "hyena/2": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-21-36-614857"
        ),
        "lstm/1": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/12-00-28-036885"
        ),
        "transformer/12": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-222033"
        ),
        "linear_transformer/8": (
            "/raid/lingo/akyurek/git/iclmodels/outputs/2023-11-15/11-44-53-201063"
        ),
    }

    data = get_results(
        os.path.join(exp_folders["transformer/8"], "generations", "200_test.txt")
    )[:20]
    n3gramprobs = get_ngram_probs(
        data, ngram=3, uniform=False, backoff=True, addone=False
    )
    dfaprobs = get_dfa_probs(data)

    print(get_greedy_dfa_accuracy(n3gramprobs, dfaprobs))
    print(get_cross_entropy(n3gramprobs, dfaprobs))
