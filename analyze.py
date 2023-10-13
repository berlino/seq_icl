from typing import List
import os
import dataclasses
import glob
import pandas as pd
import numpy as np
import pickle
from src.dataloaders.dfa import DFA

def eval_dfa(dfa_str):
    # find where ", rng" is
    rng_idx = dfa_str.find(", rng")
    dfa_str = dfa_str[:rng_idx] + ", rng=np.random.default_rng(0))"
    return eval(dfa_str)

def get_transition_info(row):
    input, target, pred, dfa = row[["input", "target", "pred", "dfa"]]
    preds = pred.split("|")
    transition_states = []
    for index, example in enumerate(input.split("|")):
        pred = preds[index]
        transitions = ""
        for t in range(len(example)):
            if index == 0:
                if t < len(pred) and example[t] != "|":
                    current_word = " ".join(list(example[:t+1] + pred[t]))
                    transitions += str(int(dfa(current_word)))
            else:
                current_word = " ".join(list(example[:t] + pred[t]))
                transitions += str(int(dfa(current_word)))

        transition_states.append(transitions)
    transition_states = "|".join(transition_states)
    total = np.sum(list(map(int, list(transition_states.replace("|", "")))))
    return transition_states, total, len(transition_states.replace("|", ""))

def get_uniform_probs(chars, vocab):
    probs = np.zeros(len(vocab))
    for c in chars:
        probs[vocab.get_id(c)] = 1 / len(chars)
    return probs

def get_dfa_probs(input, dfa, vocab):
    probs = []
    for index, example in enumerate(input.split("|")):
        for t in range(0, len(example)+1):
            if t == 0 and index == 0:
                continue
            current_word = " ".join(list(example[:t]))
            node = dfa.forward(current_word)
            possibilities = list(dfa.transitions[node].keys())
            probs.append(get_uniform_probs(possibilities,  vocab))
    return np.array(probs)

def get_traces(row):
    input, target, pred, dfa = row[["input", "target", "pred", "dfa"]]
    input_states = []
    pred_states = []
    pred_labels = []
    preds = pred.split("|")
    for index, example in enumerate(input.split("|")):
        if index == 0 or len(example) == 0:
            continue
        states = dfa.trace(" ".join(list(example)))
        states = list(map(str, states))
        input_states.append("".join(states))
        if index < len(preds):
            pred = example[0] + preds[index][1:]
            pred = " ".join(list(pred))
            # dfa label
            label = str(dfa(pred))
            pred_labels.append(label)
            states = dfa.trace(pred)
            states = list(map(str, states))
            pred_states.append("".join(states))
    input_states = "|".join(input_states)
    pred_states = "|".join(pred_states)
    pred_labels = "|".join(pred_labels)
    return input_states, pred_states, pred_labels

@dataclasses.dataclass
class Probs:
    probs: np.ndarray
    vocab: List

def get_results(exp_folder):
    generation_files = glob.glob(exp_folder + "/*_test.txt")
    results = []
    for file in generation_files:
        basename = os.path.basename(file)
        basename = basename.replace("_test.txt", "")
        fileid = int(basename)
        df = pd.read_csv(file, sep="\t", header=None, names=["input", "target", "pred", "dfa", "diff_n_gram", "diff_dfa", "diff_dfa_ngram"])
        df["dfa"] = df["dfa"].apply(lambda x: eval_dfa(x))
        pkl_file = file.replace("txt", "pkl")
        probs = None
        if os.path.isfile(pkl_file):
            with open(pkl_file, "rb") as f:
                probs = pickle.load(f)
        for index, row in df.iterrows():
            input_states, pred_states, pred_labels = get_traces(row)
            df.loc[index, "input_states"] = input_states
            df.loc[index, "pred_states"] = pred_states
            df.loc[index, "pred_labels"] = pred_labels
            if probs is not None:
                df.loc[index, "probs"] = Probs(probs["probs"][index], probs["vocab"])
        results.append(df)
    total_acc = 0.0
    total = 0.0
    if len(results) > 2:
        for index, row in results[2].iterrows():
            transitions, acc, length = get_transition_info(row)
            total_acc += acc
            total += length

        print(total_acc / total)
    return results, generation_files



def pretty_print(example):
    input, target, pred, dfa,input_states, pred_states, pred_labels = example
    transitions, acc, length= get_transition_info(example)
    print("DFA: ", dfa)
    print("Input: ", input)
    print("Pred: ", pred)
    print("Target: ", target)
    print("Input states: ", input_states)
    print("Pred states: ", pred_states)
    print("Pred labels: ", pred_labels)
    print("Transitions: ", transitions)
    print()


if __name__ == "__main__":

    #exp_folder_tf = "outputs/2023-09-06/02-42-23-025200/generations" # TF
    exp_folder_tf = "outputs/2023-09-19/10-57-28-816861/generations"
    # exp_folder_lstm = "outputs/2023-09-07/02-22-14-205796/generations" # LSTM
    exp_folder_lstm = "outputs/2023-09-19/11-05-57-447748/generations"
    tf_results, tf_files = get_results(exp_folder_tf)
    lstm_results, lstm_files = get_results(exp_folder_lstm)
    pretty_print(tf_results[2].iloc[7])
    pretty_print(lstm_results[2].iloc[6])
    total = 0.0
    corrects = 0.0
    for index, example in lstm_results[2].iterrows():
        lstm_pred = str(example["pred"])
        example = example[["input", "target", "pred", "dfa"]].copy()
        transitions, correct, length = get_transition_info(example)
        corrects += correct
        total += length
    print(corrects / total)