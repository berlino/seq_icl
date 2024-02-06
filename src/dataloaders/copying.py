'''Synthetic datasets to test in-context learning ability.'''

import os
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from typing import Dict
import numpy as np
from tqdm import tqdm
from collections import Counter

from src.dataloaders.base import SequenceDataset

class Vocab:
    """Custom vocab."""
    def __init__(self, vocab_size: int, special_vocabs: Dict):
        # Special tokens hold copy_prefix and noop/pad token etc
        assert "copy_prefix" in special_vocabs
        self.special_vocabs = special_vocabs
        vocab = [str(v) for v in list(range(vocab_size))]
        self.non_special_vocab = sorted(list(vocab))
        self.vocab = sorted(list(set(vocab + list(self.special_vocabs.values()))))
        self.v2id = {v:i for i,v in enumerate(self.vocab)}
        self.vocab_size = len(vocab)

    def get_next_vocab(self, token: str):
        """Gets next token excluding special_vocabs."""
        id = (self.get_id(token) + 1) % self.vocab_size
        while self.get_vocab(id) in self.special_vocabs:
            id = (id + 1) % self.vocab_size
        return self.get_vocab(id)

    @property
    def copy_prefix(self):
        return self.special_vocabs["copy_prefix"]

    @property
    def start_prefix(self):
        return self.special_vocabs["start_prefix"]

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

    def tokenize(self, text: str, return_tensor=False, mask_input=False):
        input_ids = [self.vocab.get_id(t) for t in text.split()]
        if self.vocab.get_id(self.vocab.copy_prefix) not in input_ids:
            raise ValueError("Input text must contain copy_prefix token.")
        copy_prefix_pos = input_ids.index(self.vocab.get_id(self.vocab.copy_prefix))
        labels = input_ids
        if mask_input:
            # Mask the input tokens for loss but do not mask the copied token
            labels = [-100] * (copy_prefix_pos+1) + labels[copy_prefix_pos+1:]
        if return_tensor:
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)

        return (input_ids, labels)

    def decode(self, ids: list):
        return " ".join([self.vocab.get_vocab(id) for id in ids])

def generate_copy(
    vocab: Vocab,
    input_seq_len: int,
    rng: np.random.Generator,
    valid_chars: list = None,
):
    """Generate sequence where the copy prefix is inserted into the input
    and then the character after the copy prefix is copied at the end.
    """
    vocab_seq = rng.choice(
        vocab.vocab,
        input_seq_len,
        replace=True,
        # Do not generate any special tokens
        p=[1/(len(vocab)-len(vocab.special_tokens)) if p not in vocab.special_tokens else 0 for p in vocab.vocab])
    vocab_seq = vocab_seq.tolist()
    vocab_seq  = [vocab.start_prefix] + vocab_seq + [vocab.copy_prefix] + vocab_seq
    if valid_chars is not None:
        raise NotImplementedError("Valid chars not implemented for induction heads.")
    return " ".join(vocab_seq)


class CopyDataModule(SequenceDataset):
    _name_ = "copying"

    def __init__(
        self,
        num_examples: int,
        num_test_examples: int,
        vocab_size: int,
        input_seq_len: int,
        seed: int = 0,
        batch_size: int = 32,
        split_train_test: bool = False,
        test_seq_len: int = None,
        data_dir: str = None,
        *args, **kwargs
    ):
        self.num_examples = num_examples
        self.num_test_examples = num_test_examples
        self.input_seq_len = input_seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        self.batch_size = batch_size
        self.split_train_test = split_train_test # let the same copy chars appear in train/test
        self.data_dir = data_dir

        if test_seq_len is not None:
            self.test_seq_len = test_seq_len
        else:
            self.test_seq_len = input_seq_len

        special_vocabs = {
            "copy_prefix": "=>",
            "start_prefix": "<s>",
            "noop": "."
        }

        self.special_vocabs = special_vocabs
        self.vocab = Vocab(vocab_size-len(special_vocabs), special_vocabs=special_vocabs)
        self.tokenizer = Tokenizer(self.vocab)

        self.num_extra_seq_len = 3

        self.total_seq_len = 2 * max(self.input_seq_len, self.test_seq_len) + self.num_extra_seq_len

    def generate_example(self, seqlen=None, valid_chars=None, mask_input=False):
        vocab_seq = generate_copy(vocab=self.vocab, rng=self.rng, input_seq_len=seqlen, valid_chars=valid_chars)
        return self.tokenizer.tokenize(vocab_seq, return_tensor=True, mask_input=mask_input)

    def setup(self, stage=None):
        train_tensor = test_tensor = None
        if self.data_dir is not None:
            try:
                train_tensor = torch.load(os.path.join(self.data_dir,
                    f"train_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt"))
                test_tensor = torch.load(os.path.join(self.data_dir,
                    f"test_{self.copy_method}_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt"))
            except:
                pass

        if train_tensor is None or test_tensor is None:
            if hasattr(self, 'dataset'):
                return
            self.rng = np.random.default_rng(self.seed)

            if self.split_train_test:
                all_vocab = self.vocab.non_special_vocab
                train_vocab = set(self.rng.choice(all_vocab, size=len(all_vocab) // 2, replace=False))
                test_vocab = set(all_vocab) - train_vocab
                train_vocab = list(train_vocab)
                test_vocab = list(test_vocab)
            else:
                train_vocab = None
                test_vocab = None

            all_examples = []
            for i, (example_count, valid_vocab) in enumerate(zip([self.num_examples, self.num_test_examples], [train_vocab, test_vocab])):
                examples = []
                while len(examples) < example_count:
                    input_ids, target_ids = self.generate_example(
                            seqlen=self.input_seq_len if i == 0 else self.test_seq_len,
                            valid_chars=valid_vocab,
                            mask_input=True,
                        )
                    examples.append((input_ids, target_ids))

                self.rng.shuffle(examples)
                all_examples.append(examples)

            # all_examples = torch.concat(all_examples)
            train_tensor = torch.stack([torch.stack([example[:-1], label[1:]]) for example, label in all_examples[0]])
            test_tensor = torch.stack([torch.stack([example[:-1], label[1:]]) for example, label in all_examples[1]])

            if self.data_dir is not None:
                torch.save(train_tensor, os.path.join(self.data_dir,
                    f"train_copy_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt")
                )
                torch.save(test_tensor, os.path.join(self.data_dir,
                    f"test_copy_{self.num_examples}_{self.vocab_size}_{self.input_seq_len}.pt")
                )

        self.dataset = {
            'train': TensorDataset(train_tensor[:, 0, :], train_tensor[:, 1, :]),
            'test': TensorDataset(test_tensor[:, 0, :], test_tensor[:, 1, :])
        }

    def train_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset['train'], shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset['test'], shuffle=False)

    def test_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset['test'], shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=shuffle,
            persistent_workers=True
        )