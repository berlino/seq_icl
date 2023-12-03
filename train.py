import copy
import os
import random
import time
import dataclasses
from functools import partial, wraps
from typing import Callable, List, Sequence
import pickle
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from tqdm.auto import tqdm

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks

log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

from analyze import get_dfa_probs
from ngram import (
    predict_with_n_gram_back_off,
    prob_distance,
    prob_distance_dfa,
    prob_distance_dfa_ngram,
)


@dataclasses.dataclass
class Probs:
    probs: np.ndarray
    vocab: List


# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
        .. code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created"
                    " instances of `WandbLogger` will reuse this run. If this is not"
                    " desired, call `wandb.finish()` before instantiating"
                    " `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                print("Here, we are in the attach")
                print("attach_id: ", attach_id)
                self._experiment = wandb._attach(attach_id)
                print("self._experiment: ", self._experiment)
            else:
                # create new wandb process
                print("Here, we are in the custom wandb logger.")
                print("self._wandb_init: ", self._wandb_init)
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        print("self._experiment: ", self._experiment)
                        break
                    except Exception as e:
                        print("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric(
                        "*", step_metric="trainer/global_step", step_sync=True
                    )

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level, so can access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        self.setup()  ## Added by KS

    def setup(self, stage=None):
        if not self.hparams.train.disable_dataset:
            self.dataset.setup()

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more memory than the others
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        self.model = utils.instantiate(registry.model, self.hparams.model)

        if (name := self.hparams.train.post_init_hook["_name_"]) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs["_name_"]
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # Instantiate the task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )
        # Extract the modules so they show up in the top level parameter count
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)

        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, "loss_val"):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics

        self.final_val_torchmetrics = self.task.final_val_torchmetrics
        self.final_test_torchmetrics = self.task.final_test_torchmetrics

        os.makedirs("samples", exist_ok=True)

        if "dfa" in self.hparams["dataset"]["_name_"]:
            for name, dataloader in zip(*self._eval_dataloaders()):
                dataset = dataloader.dataset
                tokenizer = dataset.tokenizer
                with open(f"samples/{name}.txt", "w") as f:
                    for index in range(len(dataset)):
                        data = dataset[index]
                        x, y, dfa = data
                        print("".join(tokenizer.decode(x)).replace(".", ""), file=f)
            train_dataloader = self.train_dataloader()
            dataset = train_dataloader.dataset


            tokenizer = dataset.tokenizer

            # with open(f"samples/train.txt", "w") as f:
            #     for index in range(len(dataset)):
            #         data = dataset[index]
            #         x, y, dfa = data
            #         print("".join(tokenizer.decode(x)).replace(".", ""), file=f)

    def load_state_dict(self, state_dict, strict=True):
        if self.hparams.train.pretrained_model_state_hook["_name_"] is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            # Modify the checkpoint['state_dict'] inside model_state_hook e.g. to inflate 2D convs to 3D convs
            state_dict = model_state_hook(self.model, state_dict)

        print("Custom load_state_dict function is running.")

        # note, it needs to return something from the normal function we overrided
        return super().load_state_dict(state_dict, strict=strict)

    def _check_config(self):
        assert self.hparams.train.state.mode in [
            None,
            "none",
            "null",
            "reset",
            "bptt",
            "tbptt",
        ]
        assert (
            (n := self.hparams.train.state.n_context) is None
            or isinstance(n, int)
            and n >= 0
        )
        assert (
            (n := self.hparams.train.state.n_context_eval) is None
            or isinstance(n, int)
            and n >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, train=True):
        """Handle logic for state context."""
        # Number of context steps
        key = "n_context" if train else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ["tbptt"]:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == "tbptt":
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    # def forward(self, batch):
    #     """Passes a batch through the encoder, backbone, and decoder"""
    #     # z holds arguments such as sequence length
    #     x, y, *z = batch # z holds extra dataloader info such as resolution
    #     if len(z) == 0:
    #         z = {}
    #     else:
    #         assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
    #         z = z[0]

    #     x, w = self.encoder(x, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
    #     x, state = self.model(x, **w, state=self._state)
    #     self._state = state
    #     x, w = self.decoder(x, state=state, **z)
    #     return x, y, w

    def forward(self, batch, return_hidden_outputs=False):
        return self.task.forward(
            batch,
            self.encoder,
            self.model,
            self.decoder,
            self._state,
            return_hidden_outputs=return_hidden_outputs,
        )

    def step(self, x_t):
        x_t, *_ = self.encoder(
            x_t
        )  # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        # x_t = x_t[:, None, ...] # Dummy length
        # x_t, *_ = self.decoder(x_t, state=state)
        # x_t = x_t[:, 0, ...]
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def _get_dfa_accuracy(self, x, y, batch, dfas):
        preds = x.argmax(dim=-1).detach().cpu().numpy()
        inputs = batch[0].detach().cpu().numpy()
        char_labels = []
        total = 0.0
        correct = 0.0
        for b in range(preds.shape[0]):
            current_labels = []
            pred_chars = [
                self.task.dataset.vocab.get_vocab(token) for token in preds[b]
            ]
            input_chars = [
                self.task.dataset.vocab.get_vocab(token)
                for token in inputs[b]
                if self.task.dataset.vocab.get_vocab(token) != "."
            ]
            dfa = dfas[b]
            for t in range(len(input_chars)):
                if len(input_chars) > t + 1:
                    if input_chars[t + 1] == "|":
                        continue
                    if input_chars[t + 1] == ".":
                        break
                if len(pred_chars) > t:
                    current_chars = input_chars[: t + 1] + [pred_chars[t]]
                    # take the last example
                    current_word = " ".join(current_chars).split(" | ")[-1]
                    label = int(dfa(current_word))
                    if current_word:
                        current_labels.append(label)
                        total += 1
                        correct += label
                else:
                    print("preds are shorter than inputs")
                    current_labels.append(0)
                    total += 1
            char_labels.append(current_labels)
        # get the accuracy
        return char_labels, correct / total

    def _writes_to_file(self, prefix, x, y, batch, dfas, ngram=3, hidden_outputs=None, char_labels=None):
        inputs = batch[0].detach().cpu().numpy()
        targets = y.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        preds = x.argmax(axis=-1)
        os.makedirs("generations", exist_ok=True)
        os.makedirs(f"generations/{self.current_epoch}_{prefix}_batch", exist_ok=True)
        # print(os.getcwd())
        attention_scores = None
        if hidden_outputs is not None:
            # check if hidden_outputs is a tuple
            if isinstance(hidden_outputs, tuple):
                hidden_outputs, attention_scores = hidden_outputs

            for i in range(200):

                path = f"generations/{self.current_epoch}_{prefix}_batch/{i}.pkl"

                if not os.path.isfile(path):

                    if hidden_outputs is not None:
                        saved_hidden_outputs = [
                            hidden_output.cpu().numpy() for hidden_output in hidden_outputs
                        ]
                    else:
                        saved_hidden_outputs = None

                    if attention_scores is not None:
                        saved_attention_scores = [
                            attention_score.cpu().numpy() for attention_score in attention_scores
                        ]
                    else:
                        saved_attention_scores = None

                    with open(path, "wb") as handle:
                        pickle.dump(
                            {
                                "probs": x,
                                "dfas": dfas,
                                "char_labels": char_labels,
                                "vocab": self.task.dataset.vocab.vocab,
                                "hidden_outputs": saved_hidden_outputs,
                                "attention_scores": saved_attention_scores,
                            },
                            handle,
                        )
                    break

        total_l1_chars = 0.0
        total_l1_model_dfa = 0.0
        total_l1_model_ngram = 0.0
        total_l1_dfa_ngram = 0.0
        total_n_gram_chars = 0.0
        total_n_gram_loss = 0.0
        total_n_gram_corrects = 0.0

        with open(f"generations/{self.current_epoch}_{prefix}.txt", "a+") as handle:
            for b in range(inputs.shape[0]):
                pred_chars = []
                target_chars = []
                for t in range(len(targets[b])):
                    pred_char = self.task.dataset.vocab.get_vocab(preds[b][t])
                    if targets[b][t] == -100:
                        if t + 1 < len(targets[b]):
                            if targets[b][t + 1] == -100:
                                break
                            else:
                                target_char = "|"
                                pred_char = "|"
                        else:
                            break
                    else:
                        target_char = self.task.dataset.vocab.get_vocab(targets[b][t])

                    pred_chars.append(pred_char)
                    target_chars.append(target_char)

                pred = "".join(pred_chars)
                target = "".join(target_chars)

                input_chars = [
                    self.task.dataset.vocab.get_vocab(token)
                    for token in inputs[b]
                    if token != -100
                ]
                input_chars = [char for char in input_chars if char != "."]
                input = "".join(input_chars)
                dfa = str(dfas[b])

                model_probs = torch.softmax(torch.tensor(x[b]), dim=-1).detach().cpu().numpy()
                dfa_probs = get_dfa_probs(input, dfas[b], vocab=self.task.dataset.vocab)
                model_probs = model_probs[:len(dfa_probs)]
                total_l1_model_dfa += abs(model_probs - dfa_probs).sum()
                total_l1_chars += dfa_probs.shape[0]
                if ngram != -1:
                    n_gram_probs = predict_with_n_gram_back_off(input, N=ngram, global_vocab=self.task.dataset.vocab)
                    total_l1_model_ngram += abs(model_probs - n_gram_probs).sum()
                    total_l1_dfa_ngram += abs(dfa_probs - n_gram_probs).sum()

                    for t in range(len(targets[b])):
                        if targets[b][t] == -100:
                            if t + 1 < len(targets[b]):
                                if targets[b][t + 1] == -100:
                                    break
                                else:
                                    continue
                            else:
                                break
                        else:
                            total_n_gram_loss -= np.log(n_gram_probs[t][targets[b][t]] + 1e-5)
                            total_n_gram_chars += 1
                            if dfa_probs[t, n_gram_probs[t].argmax()] != 0.0:
                                total_n_gram_corrects += 1


                print(
                    f"{input}\t{target}\t{pred}",
                    file=handle,
                )

        return (
            total_l1_model_ngram / total_l1_chars,
            total_l1_model_dfa / total_l1_chars,
            total_l1_dfa_ngram / total_l1_chars,
            total_n_gram_loss / total_n_gram_chars,
            total_n_gram_corrects / total_n_gram_chars,

        )

    def _shared_step(self, batch, batch_idx, prefix="train"):
        metric_prefix = prefix.replace("final/", "final_")

        prefix = prefix.replace("final/", "")
        if ("final" in  metric_prefix) and (prefix == "test" or prefix == "val"):
            return_hidden_outputs = True
        else:
            return_hidden_outputs = False

        return_hidden_outputs = False

        self._process_state(batch, batch_idx, train=(prefix == "train"))
        x, y, w = self.forward(batch, return_hidden_outputs=return_hidden_outputs)

        if "dfas" in w:
            char_labels, dfa_accuracy = self._get_dfa_accuracy(x, y, batch, w["dfas"])
            # write to a file
            hidden_outputs = w["hidden_outputs"] if return_hidden_outputs else None

            if ("final" in metric_prefix) or return_hidden_outputs:
                model_ngram_diff, model_dfa_diff, dfa_ngram_diff, n_gram_loss, n_gram_dfa_acc = self._writes_to_file(
                    prefix, x, y, batch, w["dfas"],
                    ngram=3,
                    hidden_outputs=hidden_outputs,
                    char_labels=char_labels,
                )
            # elif self.current_epoch % 50 == 1:
            #     n_gram_diff, dfa_diff, dfa_ngram_diff = self._writes_to_file(prefix, x, y, batch, w["dfas"])

        # Loss
        x = rearrange(x, "... C -> (...) C")
        y = rearrange(y, "... -> (...)")

        if prefix == "train":
            loss = self.loss(x, y, **w)
        else:
            loss = self.loss_val(x, y, **w)

        # Metrics
        metrics = self.metrics(x, y, **w)
        metrics["loss"] = loss
        if prefix != "train" and "dfas" in w:
            metrics["dfa_accuracy"] = dfa_accuracy
            if ("final" in metric_prefix) or return_hidden_outputs:
                metrics["model_ngram_diff"] = model_ngram_diff
                metrics["model_dfa_diff"] = model_dfa_diff
                metrics["dfa_ngram_diff"] = dfa_ngram_diff
                metrics["n_gram_loss"] = n_gram_loss
                metrics["n_gram_dfa_acc"] = n_gram_dfa_acc

        metrics = {f"{metric_prefix}/{k}": v for k, v in metrics.items()}

        log_on_step = (
            "eval" in self.hparams
            and self.hparams.eval.get("log_on_step", False)
            and prefix == "train"
        )

        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        #  Calculate torchmetrics
        torchmetrics = getattr(self, f"{metric_prefix}_torchmetrics")
        torchmetrics(x, y, loss=loss)

        # log the whole dict, otherwise lightning takes the mean to reduce it
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        self.log_dict(
            torchmetrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")

    def training_epoch_end(self, outputs):
        # Log training torchmetrics
        super().training_epoch_end(outputs)
        # self.log_dict(
        #     {f"train/{k}": v for k, v in self.task.get_torchmetrics("train").items()},
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     add_dataloader_idx=False,
        #     sync_dist=True,
        # )

    def on_validation_epoch_start(self):
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)

    def validation_epoch_end(self, outputs):
        # Log all validation torchmetrics
        super().validation_epoch_end(outputs)
        # for name in self.val_loader_names:
        #     self.log_dict(
        #         {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         add_dataloader_idx=False,
        #         sync_dist=True,
        #     )

    def on_test_epoch_start(self):
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    def test_epoch_end(self, outputs):
        # Log all test torchmetrics
        super().test_epoch_end(outputs)
        # for name in self.test_loader_names:
        #     self.log_dict(
        #         {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         add_dataloader_idx=False,
        #         sync_dist=True,
        #     )

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log the loss explicitly so it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Log any extra info that the models want to expose (e.g. output norms)
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ema = (
            self.val_loader_names[dataloader_idx].endswith("/ema")
            and self.optimizers().optimizer.stepped
        )  # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial sanity check
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):
        # Set zero weight decay for some params
        if "optimizer_param_grouping" in self.hparams.train:
            add_optimizer_hooks(
                self.model, **self.hparams.train.optimizer_param_grouping
            )

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = utils.instantiate(
            registry.optimizer, self.hparams.optimizer, params
        )

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s)
            for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups", hps)
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        ### Layer Decay ###

        if self.hparams.train.layer_decay["_name_"] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay["_name_"],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        "params": [],
                        "lr": None,
                        "weight_decay": self.hparams.optimizer.weight_decay,
                    }
                layer_wise_groups[layer_id]["params"].append(p)

                if layer_id > num_max_layers:
                    num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group["lr"] = self.hparams.optimizer.lr * (
                    self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id)
                )

            # Reset the torch optimizer's param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)
        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        # test_loaders = self.dataset.train_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (eg if test is duplicate)
        if self.hparams.train.get("remove_test_loader_in_eval", None) is not None:
            return val_loader_names, val_loaders
        # default behavior is to add test loaders in eval
        else:
            return val_loader_names + test_loader_names, val_loaders + test_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in test_loader_names]
        return test_loaders


### pytorch-lightning utils and entrypoint ###


def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups
        import wandb

        logger = WandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Add ProgressiveResizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        print(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            # Stage params are resolution and epochs, pretty print
            print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Configure ddp automatically
    n_devices = config.trainer.get("devices", 1)
    if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get("strategy", None) is None:
        config.trainer.strategy = dict(
            _target_="pytorch_lightning.strategies.DDPStrategy",
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
        )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )

    return trainer


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    # Run initial validation epoch (useful for debugging, finetuning)
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(model)

    if config.train.ckpt is not None:
        # trainer.fit(model, ckpt_path=config.train.ckpt)
        pass
    else:
        trainer.fit(model)
    if config.train.test:
        trainer.test(model, ckpt_path="best")


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)

    # Pretty print config using Rich library
    utils.train.print_config(config, resolve=True)
    train(config)


if __name__ == "__main__":
    main()
