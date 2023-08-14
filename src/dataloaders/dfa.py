"""Synthetic datasets to test in-context learning ability."""
from typing import Tuple
import os
import torch
import dataclasses
from torch.utils.data import TensorDataset, Dataset, DataLoader
from typing import Dict
import numpy as np
from tqdm import tqdm
from collections import Counter

from src.dataloaders.base import SequenceDataset


@dataclasses.dataclass
class DFA:
    """Represents a DFA"""
    num_nodes: int
    alphabet: Tuple[str]
    transitions: Tuple[dict]
    rng: np.random.Generator

    def __hash__(self):
        # hash transitions
        transitions = tuple(
            tuple(sorted(transition.items())) for transition in self.transitions
        )
        return hash((self.num_nodes, self.alphabet, transitions))

    def __post_init__(self):
        assert len(self.transitions) == self.num_nodes

    def __call__(self, word: str):
        current_node = 0
        for symbol in word:
            if symbol not in self.transitions[current_node]:
                return False
            else:
                current_node = self.transitions[current_node][symbol]
        return True

    def sample(self, length=1):
        """Samples a random word from the DFA"""
        current_node = 0
        word = ""
        for _ in range(length):
            outgoing_symbols = list(self.transitions[current_node].keys())
            symbol = self.rng.choice(outgoing_symbols)
            word += symbol
            current_node = self.transitions[current_node][symbol]
        return word


class RandomDFASampler:
    """Samples random DFAs given configs"""

    num_nodes: int
    alphabet: Tuple[str]
    max_outgoing_edge: int
    rng: np.random.Generator = None

    def __init__(
        self,
        num_nodes: int,
        alphabet: Tuple[str],
        max_outgoing_edge: int,
        seed: int = 42,
    ):
        self.num_nodes = num_nodes
        self.alphabet = alphabet
        self.max_outgoing_edge = max_outgoing_edge
        self.rng = np.random.default_rng(seed)

    def sample(self):
        transitions = [{} for _ in range(self.num_nodes)]
        for node in range(self.num_nodes):
            num_transitions = self.rng.integers(1, self.max_outgoing_edge)
            transition_symbols = self.rng.choice(
                self.alphabet, size=num_transitions, replace=False
            )
            transition_nodes = self.rng.integers(self.num_nodes, size=num_transitions)
            transitions[node] = dict(zip(transition_symbols, transition_nodes))
        dfa_rng = np.random.default_rng(self.rng.integers(0, 2 ** 32))
        return DFA(self.num_nodes, self.alphabet, tuple(transitions), dfa_rng)


if __name__ == "__main__":
    def sample_usage():
        dfa_sampler =  RandomDFASampler(4, ("a", "b", "c", "d"), 4, seed=2)
        dfa = dfa_sampler.sample()
        word = dfa.sample(length=10)
        print(word)
        word = dfa.sample(length=10)
        print(word)

    sample_usage()

class Vocab:
    """Custom vocab."""

    def __init__(self, vocab_size: int, special_vocabs: Dict):
        # Special tokens hold seperator and noop/pad token etc
        self.special_vocabs = special_vocabs
        vocab = [str(v) for v in list(range(vocab_size))]
        self.non_special_vocab = sorted(list(vocab))
        self.vocab = sorted(list(set(vocab + list(self.special_vocabs.values()))))
        self.v2id = {v: i for i, v in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def get_next_vocab(self, token: str):
        """Gets next token excluding special_vocabs."""
        id = (self.get_id(token) + 1) % self.vocab_size
        while self.get_vocab(id) in self.special_vocabs:
            id = (id + 1) % self.vocab_size
        return self.get_vocab(id)

    @property
    def seperator(self):
        return self.special_vocabs["seperator"]

    @property
    def noop(self):
        return self.special_vocabs["noop"]

    @property
    def special_tokens(self):
        return set(self.special_vocabs.values())

    def get_id(self, token: str):
        return self.v2id[token]

    def get_vocab(self, id: int):
        return self.vocab[id]

    def __len__(self):
        return len(self.vocab)


class Tokenizer:
    """Custom Tokenizer for our own vocab."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def tokenize(self, text: str, return_tensor: bool = False, mask_input: bool = False):
        input_ids = [self.vocab.get_id(t) for t in text.split()]
        labels = input_ids

        if return_tensor:
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def decode(self, ids: list):
        return " ".join([self.vocab.get_vocab(id) for id in ids])


class ICLDFADataModule(SequenceDataset):
    _name_ = "icl_dfa"

    def __init__(
        self,
        num_examples: int,
        num_test_examples: int,
        vocab_size: int,
        max_num_nodes: int,
        max_num_in_context_examples: int,
        min_num_in_context_examples: int,
        max_outgoing_edges: int,
        max_len_per_example: int,
        number_duplicates_per_epoch: int = 0,
        input_seq_len: int = 1024,
        seed: int = 0,
        batch_size: int = 32,
        split_train_test: bool = False,
        data_dir: str = None,
        *args,
        **kwargs,
    ):
        self.num_examples = num_examples
        self.num_test_examples = num_test_examples
        self.vocab_size = vocab_size
        self.number_duplicates_per_epoch = number_duplicates_per_epoch

        self.batch_size = batch_size
        self.split_train_test = (
            split_train_test  # let the same copy chars appear in train/test
        )
        self.data_dir = data_dir
        self.max_num_nodes = max_num_nodes
        self.max_num_in_context_examples = max_num_in_context_examples
        self.min_num_in_context_examples = min_num_in_context_examples
        self.max_outgoing_edges = max_outgoing_edges
        self.max_len_per_example = max_len_per_example
        self.input_seq_len = input_seq_len
        self.seed = seed

        special_vocabs = {"seperator": "|", "noop": "."}
        self.special_vocabs = special_vocabs
        self.vocab = Vocab(
            vocab_size-2, special_vocabs=special_vocabs
        )
        self.tokenizer = Tokenizer(self.vocab)

    def generate_example(self, dfa: DFA, num_examples: int):
        example = ""
        for _ in range(num_examples):
            length = self.rng.integers(self.max_len_per_example)
            word = dfa.sample(length=length)
            example += word + "|"
        example = example[:-1]
        if len(example) > self.input_seq_len:
            example = example[:self.input_seq_len]
        example = " ".join(list(example))  # separate chars with space
        return self.tokenizer.tokenize(example, return_tensor=True)

    def setup(self, stage=None):
        train_tensor = test_tensor = None
        if self.data_dir is not None:
            try:
                train_tensor = torch.load(
                    os.path.join(
                        self.data_dir,
                        f"train_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt",
                    )
                )
                test_tensor = torch.load(
                    os.path.join(
                        self.data_dir,
                        f"test_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt",
                    )
                )
            except:
                pass

        if train_tensor is None or test_tensor is None:
            if hasattr(self, "dataset"):
                return
            self.rng = np.random.default_rng(self.seed)

            DFAs = set([])
            for _ in range(self.num_examples * 10):
                num_nodes = self.rng.integers(2, self.max_num_nodes)
                alphabet = self.rng.choice(self.vocab_size, size=num_nodes, replace=False)
                alphabet = tuple((str(a) for a in alphabet))
                sampler = RandomDFASampler(
                        num_nodes,
                        alphabet,
                        min(self.max_outgoing_edges, len(alphabet)),
                    )
                sampler.rng = self.rng
                dfa = sampler.sample()
                DFAs.add(dfa)
                if len(DFAs) >= self.num_examples + self.num_test_examples:
                    break
            DFAs = list(DFAs)
            self.rng.shuffle(DFAs)
            if len(DFAs) < self.num_examples + self.num_test_examples:
                print(
                    "Warning: not enough unique DFAs generated. Using all generated"
                    " DFAs."
                )
                # scale back
                self.num_examples = (len(DFAs) * self.num_examples) // (
                    self.num_examples + self.num_test_examples
                )
                self.num_test_examples = len(DFAs) - self.num_examples
                print(
                    f"New num_examples: {self.num_examples}, new num_test_examples:"
                    f" {self.num_test_examples}"
                )

            train_DFAs = DFAs[: self.num_examples]
            test_DFAs = DFAs[
                self.num_examples : self.num_examples + self.num_test_examples
            ]

            all_examples = []
            for DFAs in (train_DFAs, test_DFAs):
                examples = []
                for dfa in DFAs:
                    num_samples = self.rng.integers(
                        self.min_num_in_context_examples,
                        self.max_num_in_context_examples,
                    )
                    example = self.generate_example(dfa, num_samples)[
                        "input_ids"
                    ].tolist()
                    examples.append(example)

                self.rng.shuffle(examples)
                # pad examples to same length
                examples = torch.nn.utils.rnn.pad_sequence(
                    [torch.LongTensor(example) for example in examples],
                    batch_first=True,
                    padding_value=self.vocab.get_id(self.vocab.noop),
                )

                all_examples.append(examples)

            # all_examples = torch.concat(all_examples)
            train_tensor = torch.stack(
                [
                    torch.stack([example[:-1], example[1:]])
                    for example in all_examples[0]
                ]
            )
            test_tensor = torch.stack(
                [
                    torch.stack([example[:-1], example[1:]])
                    for example in all_examples[1]
                ]
            )

            if self.data_dir is not None:
                torch.save(
                    train_tensor,
                    os.path.join(
                        self.data_dir,
                        f"train_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt",
                    ),
                )
                torch.save(
                    test_tensor,
                    os.path.join(
                        self.data_dir,
                        f"test_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt",
                    ),
                )

        self.dataset = {
            "train": TensorDataset(train_tensor[:, 0, :], train_tensor[:, 1, :]),
            "test": TensorDataset(test_tensor[:, 0, :], test_tensor[:, 1, :]),
        }

    def train_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["train"], shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False)

    def test_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=10,
            shuffle=shuffle,
            persistent_workers=True,
        )


if __name__ == "__main__":
    # test dataloader
    data_module = ICLDFADataModule(
        num_examples=100,
        num_test_examples=10,
        vocab_size=10,
        max_num_nodes=10,
        max_num_in_context_examples=10,
        min_num_in_context_examples=1,
        max_outgoing_edges=4,
        max_len_per_example=10,
        seed=0,
        batch_size=32,
        split_train_test=False,
        data_dir=None,
    )

    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    for batch in tqdm(train_loader):
        print(batch)
        print(batch[0].shape)
        breakpoint()