# Baum-Welch Algorithm

import numpy as np
import time
from scipy.special import logsumexp
import torch
from torch.nn import functional as F

EPS = 1e-5
from tqdm import tqdm

def possible_states(t: int, max_states: int = 12):
    """
    Return the possible states at time t.

    Args:
        t (int): Time step.

    Returns:
        states (list): List of possible states.
    """
    if t == 0:
        return [0]
    elif t == 1:
        return [1, 2, 3, 4, 5]
    else:
        current_states = possible_states(t - 1, max_states=max_states)
        next_state = max(current_states)
        for i in range(5):
            next_state = next_state + 1
            if next_state < max_states:
                current_states.append(next_state)
        if 0 not in current_states:
            current_states.append(0)
        return current_states


def mask_transition_matrix(T, num_states: int = 144):
    # get possible states at time t
    mask = torch.ones((num_states, T), device=DEVICE) * (-9999)
    num_dfa_states = int(np.sqrt(num_states))
    qs = possible_states(0, num_dfa_states)
    for t in range(1, T + 1):
        qps = possible_states(t, num_dfa_states)
        possible_hmm_states = [q * num_dfa_states + qp for q in qs for qp in qps if q != qp]
        possible_hmm_states = torch.tensor(possible_hmm_states, device=DEVICE)
        mask[possible_hmm_states, t - 1] = 0
        qs = qps
    return mask


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def forward_backward(observations, log_A, log_B, log_pi, lengths=None, state_mask=None):
    """
    Perform the forward-backward algorithm to compute the forward and backward probabilities.

    Args:
        observations (list): List of observed sequences.
        A (np.ndarray): Transition matrix.
        B (np.ndarray): Emission matrix.
        pi (np.ndarray): Initial state probabilities.

    Returns:
        forward_probs (np.ndarray): Forward probabilities.
        backward_probs (np.ndarray): Backward probabilities.
    """
    num_states = log_A.shape[0]
    T, B = observations.shape

    forward_probs = torch.zeros((num_states, T, B), device=DEVICE)
    backward_probs = torch.zeros((num_states, T, B), device=DEVICE)

    # Forward pass

    forward_probs[:, 0, :] = log_pi[:, None] + log_B[:, observations[0]]

    log_A = log_A[:, :, None]

    for t in range(1, T):
        forward_probs[:, t, :] = log_B[:, observations[t]] + torch.logsumexp(
            forward_probs[:, None, t - 1, :] + log_A, dim=0
        )
        if state_mask is not None:
            forward_probs[:, t, :] += state_mask[:, t, None]

    # Backward pass
    time_zero_mask = torch.arange(T, device=DEVICE)[:, None] < (lengths[None, :] - 1)
    time_zero_mask = time_zero_mask[None, :, :]
    for t in range(T - 2, -1, -1):
        backward_probs[:, t, :] = (
            torch.logsumexp(
                log_A
                + log_B[None, :, observations[t + 1]]
                + backward_probs[None, :, t + 1, :],
                dim=1,
            )
            * time_zero_mask[:, t, :]
        )
        if state_mask is not None:
            backward_probs[:, t, :] += state_mask[:, t, None]

    time_mask = torch.arange(T, device=DEVICE)[:, None] >= lengths[None, :]
    time_mask = time_mask[None, :, :] * -9999

    forward_probs += time_mask
    backward_probs += time_mask

    return forward_probs, backward_probs


def get_mask_for_A(A):
    A = torch.ones_like(A, dtype=torch.bool, device=DEVICE)
    num_dfa_states = int(np.sqrt(A.shape[0]))
    for state_i in range(num_dfa_states):
        for state_j in range(num_dfa_states):
            for output_state_j in range(num_dfa_states):
                if state_i != state_j:
                    A[
                        state_i * num_dfa_states + state_j,
                        state_j * num_dfa_states + output_state_j,
                    ] = 0
    return A


def baum_welch(
    observations,
    num_states,
    num_symbols,
    max_iterations=10,
    tol=1e-4,
    A_mask=None,
    state_mask=None,
    debug=False,
):
    """
    Perform the Baum-Welch algorithm to estimate HMM parameters.

    Args:
        observations (list): List of observed sequences.
        num_states (int): Number of hidden states.
        num_symbols (int): Number of distinct symbols in the observations.
        max_iterations (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        A_hat (np.ndarray): Estimated transition matrix.
        B_hat (np.ndarray): Estimated emission matrix.
        pi_hat (np.ndarray): Estimated initial state probabilities.
    """
    num_observations = len(observations)
    A_hat = torch.rand(num_states, num_states, device=DEVICE) + EPS
    A_hat.masked_fill_(A_mask, EPS)
    A_hat /= A_hat.sum(dim=1, keepdim=True)

    B_hat = torch.rand(num_states, num_symbols, device=DEVICE) + EPS
    B_hat /= B_hat.sum(dim=1, keepdim=True)

    pi_hat = torch.ones(num_states, device=DEVICE) - EPS
    pi_hat[int(np.sqrt(num_states)) :] = EPS
    pi_hat[0] = EPS
    pi_hat /= pi_hat.sum()

    for iteration in range(max_iterations):
        log_A_hat = torch.log(A_hat)
        log_B_hat = torch.log(B_hat)
        log_pi_hat = torch.log(pi_hat)

        lengths = torch.tensor(
            [len(obs) for obs in observations], device=DEVICE, dtype=torch.long
        )

        obs = torch.nn.utils.rnn.pad_sequence(
            observations, batch_first=False, padding_value=0
        )

        T, B = obs.shape

        forward, backward = forward_backward(
            obs, log_A_hat, log_B_hat, log_pi_hat, lengths=lengths, state_mask=state_mask[:, :T]
        )

        logprob = torch.logsumexp(forward[:, lengths-1, torch.arange(B)], dim=0)

        log_gamma = forward + backward - logprob

        gamma = torch.exp(log_gamma)

        expected_pi = gamma[:, 0, :].sum(dim=-1)

        # we need to accumulate only valid time steps
        #time_mask = torch.arange(T, device=DEVICE)[:, None] < lengths[None, :]
        #time_mask = time_mask[None, :, :]
        #gamma = gamma * time_mask

        expected_B = torch.zeros((num_states, num_symbols), device=DEVICE)
        for t in range(obs.shape[0]):
            expected_B[:, obs[t]] += gamma[:, t, :]

        expected_si = gamma[:, :-1, :].sum(dim=(1, 2))

        s_ijs = (forward[:, None, :-1, :] + log_A_hat[:, :, None, None]
                + log_B_hat[None, :, obs[1:]] + backward[None, :, 1:, :]) - logprob
        s_ijs = torch.exp(s_ijs)
        # assert whether s_ij is a valid probability distribution

        # s_ijs = s_ijs * time_mask[None, :, :-1, :]
        expected_sij = s_ijs.sum(dim=(2, 3))

        # if debug:
        #     breakpoint()
        new_A_hat = expected_sij / (expected_si[:, None] + EPS)
        new_A_hat.masked_fill_(A_mask, EPS)
        new_A_hat += EPS
        new_A_hat /= new_A_hat.sum(dim=1, keepdim=True)

        new_B_hat = expected_B + EPS
        new_B_hat[:, 0] = EPS
        # new_B_hat = torch.softmax(new_B_hat / 0.1, dim=1)
        new_B_hat /= new_B_hat.sum(dim=1, keepdim=True)

        new_pi_hat = expected_pi / len(observations)
        new_pi_hat += EPS
        new_pi_hat[int(np.sqrt(num_states)) :] = EPS
        new_pi_hat[0] = EPS
        new_pi_hat /= new_pi_hat.sum()

        A_hat = new_A_hat
        B_hat = new_B_hat
        pi_hat = new_pi_hat

        # if torch.max(torch.abs(new_B_hat - B_hat)) < tol and torch.max(torch.abs(new_A_hat - A_hat)) < tol and torch.max(torch.abs(new_pi_hat - pi_hat)) < tol:
        #     break

    return torch.log(A_hat), torch.log(B_hat), torch.log(pi_hat)


def get_posterior_predictions(log_A, log_B, log_pi, observations):
    """
    Get the posterior predictions for the next observation.

    Args:
        A_hat (np.ndarray): Estimated transition matrix.
        B_hat (np.ndarray): Estimated emission matrix.
        previous_observations (list): List of observed sequences.

    Returns:
        posterior_predictions (np.ndarray): Posterior predictions.
    """
    num_states = log_A.shape[0]
    num_symbols = log_B.shape[1]

    T = len(observations)
    forward_probs = torch.zeros((num_states, T), device=DEVICE)

    forward_probs[:, 0] = log_pi + log_B[:, observations[0]]
    for t in range(1, T):
        forward_probs[:, t] = log_B[:, observations[t]] + torch.logsumexp(
            forward_probs[:, None, t - 1] + log_A, dim=0
        )

    log_joint_scores = torch.logsumexp(forward_probs[:, -1:] + log_A, dim=0)[:, None] + log_B


    # marginalize
    log_scores = torch.logsumexp(log_joint_scores, dim=0)
    # normalize
    p = torch.softmax(log_scores, dim=0)
    return p.cpu()


# Example usage:
# Define the observed sequences, number of states, and number of symbols.


def predict_with_baumwelch(inputs, vocab, max_states=12):
    num_states = max_states * max_states
    num_symbols = len(vocab)
    running_probs = []

    A_mask = get_mask_for_A(torch.zeros((num_states, num_states), device=DEVICE))
    state_mask = mask_transition_matrix(700, num_states=num_states)

    for t in tqdm(range(1, len(inputs) + 1)):
        examples = inputs[:t].split("|")

        if len(examples[-1]) == 0:
            # marginalize
            logp = torch.logsumexp(log_B + log_pi[:, None], dim=0)
            # normalize
            p = torch.softmax(logp, dim=0)
            running_probs.append(p.cpu())
            continue

        observations = []
        for example in examples:
            observations.append(
                torch.tensor(
                    [vocab.get_id(symbol) for symbol in example],
                    device=DEVICE,
                    dtype=torch.long,
                )
            )

        log_A, log_B, log_pi = baum_welch(
            observations, num_states, num_symbols, A_mask=A_mask, state_mask=state_mask, debug=t == 300
        )

        # if t == 300:
        #     breakpoint()

        posterior_predictions = get_posterior_predictions(
            log_A, log_B, log_pi, observations[-1]
        )

        running_probs.append(posterior_predictions)

    return torch.stack(running_probs).cpu().numpy()


if __name__ == "__main__":
    from probe import get_results
    from analyze import get_dfa_probs as calculate_dfa_probs
    import numpy as np

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

    def get_baumwelch_probs(results):
        vocab = Vocab(results[0]["vocab"])
        baumwelch_probs = []
        for b in range(len(results)):
            input = results[b]["input"]
            probs = predict_with_baumwelch(input, vocab, max_states=12)
            baumwelch_probs.append(probs)
        return baumwelch_probs

    def get_dfa_probs(results):
        vocab = Vocab(results[0]["vocab"])
        dfa_probs = []
        for b in range(len(results)):
            input = results[b]["input"]
            target = [vocab.get_id(t) for t in results[b]["target"]]
            probs = calculate_dfa_probs(input, results[b]["dfa"], vocab=vocab)
            dfa_probs.append(probs)
        return dfa_probs

    def get_greedy_dfa_accuracy(probs, dfa_probs):
        total = 0.0
        correct = 0.0
        for p1, pdfa in zip(probs, dfa_probs):
            indices = p1.argmax(axis=-1)[: len(pdfa)]
            correct += (pdfa[np.arange(len(pdfa)), indices] > 0).sum()
            total += len(pdfa)
        return correct / total

    exp_folders = {
        "transformer/8": "outputs/2023-10-18/11-44-08-805221",
    }

    data = get_results(exp_folders["transformer/8"] + "/generations/200_test.txt")[:1]

    vocab = Vocab(data[0]["vocab"])

    probs = {}

    probs["dfa"] = get_dfa_probs(data)
    probs["bw"] = get_baumwelch_probs(data)


    # # print(probs["dfa"][0].shape)
    # print(probs["bw"][0].shape)

    acc = get_greedy_dfa_accuracy(probs["bw"], probs["dfa"])
    print(acc)
    # probs["bw"] = [probs["bw"][0][1:]]
    # probs["dfa"] = [probs["dfa"][0][:-1]]
    # acc = get_greedy_dfa_accuracy(probs["bw"], probs["dfa"])
    # print(acc)


    breakpoint()

