import os
import glob
import pandas as pd
import numpy as np
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

def get_results(exp_folder):
    generation_files = glob.glob(exp_folder + "/*_test.txt")
    results = []
    for file in generation_files:
        basename = os.path.basename(file)
        basename = basename.replace("_test.txt", "")
        fileid = int(basename)
        df = pd.read_csv(file, sep="\t", header=None, names=["input", "target", "pred", "dfa"])
        df["dfa"] = df["dfa"].apply(lambda x: eval_dfa(x))
        for index, row in df.iterrows():
            input_states, pred_states, pred_labels = get_traces(row)
            df.loc[index, "input_states"] = input_states
            df.loc[index, "pred_states"] = pred_states
            df.loc[index, "pred_labels"] = pred_labels
        results.append(df)
    total_acc = 0.0
    total = 0.0
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

    exp_folder_tf = "outputs/2023-09-06/02-42-23-025200/generations" # TF
    exp_folder_lstm = "outputs/2023-09-07/02-22-14-205796/generations" # LSTM
    tf_results, tf_files = get_results(exp_folder_tf)
    lstm_results, lstm_files = get_results(exp_folder_lstm)
    pretty_print(tf_results[2].iloc[7])
    pretty_print(lstm_results[2].iloc[6])

    total = 0.0
    corrects = 0.0
    for index, example in lstm_results[2].iterrows():
        lstm_pred = str(example["pred"])
        example = example[["input", "target", "pred", "dfa"]].copy()
        transitions, correct, length= get_transition_info(example)
        corrects += correct
        total += length
    print(corrects / total)