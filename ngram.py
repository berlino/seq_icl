from typing import Mapping, List, Set, Tuple
import numpy as np
from collections import Counter


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


def predict_with_n_gram_back_off(inputs: str, N: int = 3) -> str:
    # inputs is in the following form  "absadf|adsfab|...."
    # N is the max n_gram order
    predictions = []
    for t in range(1, len(inputs)):
        texts = inputs[:t].split("|")
        # get counts and vocab
        counts, vocab = train_everygram(N, texts)
        # get the last N-1 chars
        prefix = texts[-1][-(N - 1) :]
        prefix = "_" * (N - 1 - len(prefix)) + prefix
        # get next char probs
        next_probs = get_next_char_prob(prefix, vocab, N, counts, vocab)
        # greedy decoding
        assert len(next_probs) == len(vocab)
        next_char_index = np.argmax(next_probs)
        if len(inputs) > t and inputs[t] == "|":
            predictions.append("|")
        else:
            predictions.append(vocab[next_char_index])
    return "".join(predictions)


if __name__ == "__main__":
    from analyze import get_results, get_transition_info
    from tqdm import tqdm

    exp_folder_tf = "outputs/2023-09-06/02-42-23-025200/generations"  # TF
    exp_folder_lstm = "outputs/2023-09-07/02-22-14-205796/generations"  # LSTM
    tf_results, tf_files = get_results(exp_folder_tf)
    lstm_results, lstm_files = get_results(exp_folder_lstm)

    corrects = 0.0
    total = 0.0
    for index, example in tqdm(tf_results[2].iterrows()):
        tf_pred = example["pred"]
        example = example[["input", "target", "pred", "dfa"]].copy()
        example["pred"] = predict_with_n_gram_back_off(example["input"], N=3)
        ngram_pred = example["pred"]
        transitions, correct, length = get_transition_info(example)
        corrects += correct
        total += length
        print(corrects / total)
