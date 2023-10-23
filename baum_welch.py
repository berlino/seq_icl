# Baum-Welch Algorithm

import numpy as np
import time
from scipy.special import logsumexp
import torch
from torch.nn import functional as F

EPS = 1e-5
from tqdm import tqdm

# def sample_dfas(max_outgoing_edges=4, max_num_nodes=12, vocab_size=20, num_samples=1000000):
#     from src.dataloaders.dfa import DFA, RandomDFASampler, Vocab
#     rng = np.random.default_rng(0)
#     DFAs = {}
#     for _ in tqdm(range(num_samples)):
#         num_nodes = rng.integers(max_outgoing_edges, max_num_nodes + 1)
#         num_alphabet = rng.integers(
#             max_outgoing_edges, vocab_size - 2 + 1
#         )
#         alphabet = rng.choice(
#             vocab_size - 2, size=num_alphabet, replace=False
#         )
#         alphabet = tuple((chr(a + 97) for a in alphabet))
#         sampler = RandomDFASampler(
#             num_nodes,
#             alphabet,
#             max_outgoing_edges,
#         )
#         sampler.rng = np.random.default_rng(rng.integers(0, 2**32))
#         dfa = sampler.sample()
#         dfa._minimize()._trim()
#         if dfa not in DFAs:
#             DFAs[dfa] = 1
#         else:
#             DFAs[dfa] += 1

#     return DFAs


def possible_states(t: int, max_states: int = 20):
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
        return [1]
    else:
        current_states = possible_states(t - 1, max_states=max_states)
        next_state = max(current_states) + 1
        if next_state < max_states:
            current_states.append(next_state)
        if 0 not in current_states:
            current_states.append(0)
        return current_states


def mask_transition_matrix(T, num_states: int = 400):
    # get possible states at time t
    mask = torch.ones((num_states, T), device=DEVICE) * (-9999)
    num_dfa_states = int(np.sqrt(num_states))
    qs = possible_states(0, num_dfa_states)
    for t in range(1, T):
        qps = possible_states(t, num_dfa_states)
        possible_hmm_states = [
            q * num_dfa_states + qp for q in qs for qp in qps
        ]
        possible_hmm_states = torch.tensor(possible_hmm_states, device=DEVICE)
        mask[possible_hmm_states, t-1] = 0
        qs = qps
    return mask

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def forward_backward(observations, log_A, log_B, log_pi):
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
    T = len(observations)

    forward_probs = torch.zeros((num_states, T), device=DEVICE)
    backward_probs = torch.zeros((num_states, T), device=DEVICE)

    # Forward pass
    forward_probs[:, 0] = log_pi + log_B[:, observations[0]]
    for t in range(1, T):
        forward_probs[:, t] = log_B[:, observations[t]] + torch.logsumexp(
            forward_probs[:, t - 1] + log_A, dim=0
        )

    # Backward pass
    backward_probs[:, -1] = 0  # log(1) = 0
    for t in range(T - 2, -1, -1):
        backward_probs[:, t] = torch.logsumexp(
            log_A + log_B[:, observations[t + 1]] + backward_probs[:, t + 1],
            dim=1,
        )

    return forward_probs, backward_probs


def get_mask_for_A(A):
    A = torch.ones_like(A, dtype=torch.bool, device=DEVICE)
    num_dfa_states = int(np.sqrt(A.shape[0]))
    for state_i in range(num_dfa_states):
        for state_j in range(num_dfa_states):
            for output_state_i in range(num_dfa_states):
                for output_state_j in range(num_dfa_states):
                    if state_j == output_state_i and  (state_i, state_j) != (output_state_i, output_state_j):
                        A[
                            state_i * num_dfa_states + state_j,
                            output_state_i * num_dfa_states + output_state_j,
                        ] = 0
    return A


def baum_welch(
    observations, num_states, num_symbols, max_iterations=100, tol=1e-4, A_mask=None, state_mask = None
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
    pi_hat[int(np.sqrt(num_states)):] = EPS
    pi_hat /= pi_hat.sum()

    for iteration in range(max_iterations):
        expected_si   = torch.zeros(num_states, device=DEVICE)
        expected_gamma_sum = torch.zeros(num_states, device=DEVICE)
        expected_sij  = torch.zeros((num_states, num_states), device=DEVICE)
        expected_sjwk = torch.zeros((num_states, num_symbols), device=DEVICE)
        expected_pi = torch.zeros(num_states, device=DEVICE)

        log_A_hat = torch.log(A_hat)
        log_B_hat = torch.log(B_hat)
        log_pi_hat = torch.log(pi_hat)

        for obs in observations:
            forward, backward = forward_backward(obs, log_A_hat, log_B_hat, log_pi_hat)

            mask = state_mask[:, :forward.shape[1]]

            forward = forward + mask
            backward = backward + mask

            logprob = (torch.logsumexp(forward[:, -1], dim=0) + torch.logsumexp(backward[:, 0], dim=0)) / 2
            loggamma = forward + backward - logprob

            gamma = torch.exp(loggamma - loggamma.max(dim=0, keepdim=True).values)
            gamma = gamma / (gamma.sum(dim=0, keepdim=True) + EPS)


            expected_si += torch.exp(torch.logsumexp(loggamma, dim=1))
            expected_pi += gamma[:, 0]
            for t in range(len(obs)):
                  expected_sjwk[:, obs[t]] += gamma[:, t]

            expected_sij += torch.exp(torch.logsumexp(
                forward[:, None, :-1]
                + log_A_hat[:, :, None]
                + log_B_hat[None, :, obs[1:]]
                + backward[None, :, 1:] - logprob, dim=-1))



        new_A_hat = expected_sij / (expected_si[:, None]  + EPS)
        new_B_hat = expected_sjwk
        new_pi_hat = expected_pi / len(observations)

        new_A_hat += EPS
        new_B_hat += EPS
        new_pi_hat += EPS

        new_A_hat.masked_fill_(A_mask, EPS)
        new_pi_hat[int(np.sqrt(num_states)):] = EPS

        new_A_hat /= new_A_hat.sum(dim=1, keepdim=True)
        new_B_hat /= new_B_hat.sum(dim=1, keepdim=True)

        new_pi_hat /= new_pi_hat.sum()

        if torch.max(torch.abs(new_A_hat - A_hat)) < tol:
            break

        A_hat = new_A_hat
        B_hat = new_B_hat
        pi_hat = new_pi_hat

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
    # Forward pass
    forward_probs[:, 0] = log_pi + log_B[:, observations[0]]
    for t in range(1, T):
        forward_probs[:, t] = log_B[:, observations[t]] + torch.logsumexp(
            forward_probs[:, t-1] + log_A, dim=0, keepdim=True
        )

    probs = forward_probs[:, -1:] + log_B
    probs = torch.logsumexp(probs, dim=0)
    probs = torch.exp(probs - probs.max())
    probs += EPS
    probs = probs / (probs.sum() + EPS)
    return probs.cpu()


# Example usage:
# Define the observed sequences, number of states, and number of symbols.


def predict_with_baumwelch(inputs, vocab, max_states=20):
    num_states = max_states * max_states
    num_symbols = len(vocab)
    running_probs = []

    A_mask = get_mask_for_A(torch.zeros((num_states, num_states), device=DEVICE))
    state_mask = mask_transition_matrix(700, num_states=num_states)

    for t in tqdm(range(1, len(inputs) + 1)):
        current_input = inputs[:t].split("|")
        if len(current_input[-1]) == 0:
            logp = torch.logsumexp(log_B + log_pi[:, None], dim=0)
            logp = logp - logp.max()
            p = torch.exp(logp) + EPS
            p = p / p.sum()
            running_probs.append(p.cpu())
            continue

        observations = []
        for input in current_input:
            observations.append(
                torch.tensor(
                    [vocab.get_id(symbol) for symbol in input],
                    device=DEVICE,
                    dtype=torch.long,
                )
            )
        t0 = time.time()
        log_A, log_B, log_pi = baum_welch(
            observations, num_states, num_symbols, A_mask=A_mask, state_mask=state_mask
        )

        posterior_predictions = get_posterior_predictions(
            log_A, log_B, log_pi, observations[-1]
        )

        running_probs.append(posterior_predictions)

    return torch.stack(running_probs).cpu().numpy()


if __name__ == "__main__":
    from probe import get_results


    # dfas = sample_dfas()
    # breakpoint()
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
            probs = predict_with_baumwelch(input, vocab, max_states=20)
            baumwelch_probs.append(probs)
        return baumwelch_probs

    exp_folders = {
        "transformer/8": "outputs/2023-10-18/11-44-08-805221",
        "transformer/2": "outputs/2023-10-18/11-44-08-874258",
        "transformer/4": "outputs/2023-10-18/11-44-08-874396",
        "transformer/1": "outputs/2023-10-18/11-44-08-882528",
        "linear_transformer/4": "outputs/2023-10-18/11-44-08-884445",
        "retnet/4": "outputs/2023-10-18/11-44-08-886024",
        "rwkv/2": "outputs/2023-10-18/11-44-08-906468",
        "h3/2": "outputs/2023-10-18/11-44-08-911000",
        "hyena/2": "outputs/2023-10-18/11-44-08-932201",
        "lstm/1": "outputs/2023-10-18/11-44-08-953521",
    }

    data = get_results(exp_folders["transformer/8"] + "/generations/200_test.txt")

    vocab = Vocab(data[0]["vocab"])

    probs = get_baumwelch_probs(data[:1])

    indices = probs[0].argmax(axis=-1)

    # get chars
    chars = [vocab.get_vocab(i) for i in indices]

    breakpoint()


# print("Estimated Transition Matrix (A):")
# print(estimated_A)
# print("Estimated Emission Matrix (B):")
# print(estimated_B)


# # print("Estimated Initial State Probabilities (pi):")
# # print(estimated_pi)
