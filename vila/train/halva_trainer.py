# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/


import os
from typing import List, Optional

import torch
from torch.utils.data import (ConcatDataset, Dataset, DistributedSampler,
                              RandomSampler, Sampler)
from transformers import PreTrainedModel, Trainer
from transformers.modeling_utils import unwrap_model
from transformers.trainer import ALL_LAYERNORM_LAYERS  # ShardedDDPOption,
from transformers.trainer import (get_parameter_names, has_length,
                                  is_sagemaker_mp_enabled, logger)
from collections import OrderedDict

import torch.nn as nn
from typing import List, Optional
from transformers import TrainerCallback, Trainer
import deepspeed
import wandb
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from collections import defaultdict
from vila.train.halva_utils import disable_dropout_in_model, PreTrainedModelWrapper, is_peft_available, is_wandb_available
import torch.nn.functional as F


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [
        lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class VILADistributedSampler(DistributedSampler):
    """This class is implemented by Jason Lu."""

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size=None,
        # NOTE: this is the total size but not per-worker
        sample_len_list=None,
        force_accumulation=True,
    ) -> None:
        import math

        import torch.distributed as dist

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True  # always True

        # NOTE: org_ is without drop last
        self.org_sample_len_list = self.per_replica_samples = sample_len_list
        assert sum(sample_len_list) == len(self.dataset)

        self.batch_size = batch_size
        self.global_batch_size = batch_size * num_replicas

        if self.drop_last:  # type: ignore[arg-type]
            self.per_replica_samples = [
                sample_len // (self.num_replicas * batch_size) * batch_size for sample_len in self.per_replica_samples
            ]
            self.num_samples = sum(self.per_replica_samples)
        else:
            raise NotImplementedError

        self.total_size = self.num_samples * self.num_replicas
        self.total_samples = [samples * self.num_replicas for samples in self.per_replica_samples]

        self.shuffle = shuffle
        self.seed = seed

        # whether to force accumulate
        self.force_accumulation = force_accumulation

    def __iter__(self):
        import random

        indices = list(range(len(self.dataset)))

        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(
                indices[sum(self.org_sample_len_list[:i]) : sum(self.org_sample_len_list[:i]) + self.total_samples[i]]
            )

        # 1. split the full indices first (note: without drop last at this moment)
        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(
                indices[sum(self.org_sample_len_list[:i]) : sum(self.org_sample_len_list[:i]) + self.total_samples[i]]
            )

        assert sum([len(indices) for indices in indices_list]) == self.total_size, (
            sum([len(indices) for indices in indices_list]),
            self.total_size,
        )

        # let's first do subsample
        for idx, indices in enumerate(indices_list):
            indices_list[idx] = indices[
                self.rank * self.per_replica_samples[idx] : (self.rank + 1) * self.per_replica_samples[idx]
            ]

        random.seed(self.seed + self.epoch)
        for indice in range(len(indices_list)):
            random.shuffle(indices_list[indice])

        indices_list = sorted(indices_list, key=lambda x: -len(x))
        all_indices = [-1] * self.num_samples
        indices_available = list(range(self.num_samples))
        for indice in indices_list:
            original_indices = range(len(indice))
            transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]
            mapped_indices = [indices_available[idx] for idx in transformed_indices]
            # update indices_available
            for idx in reversed(transformed_indices):
                del indices_available[idx]
            for i, idx in enumerate(mapped_indices):
                all_indices[idx] = indice[i]
        assert -1 not in all_indices

        return iter(all_indices)


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class LLaVATrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return VILADistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.train_batch_size,
            sample_len_list=sample_len_list,
        )

        if self.args.group_by_modality_length:
            if not isinstance(self.train_dataset, ConcatDataset):
                lengths = self.train_dataset.modality_lengths
            else:
                lengths = []
                for d in self.train_dataset.datasets:
                    lengths += d.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.eval_dataset is None or not has_length(self.eval_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.eval_sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return VILADistributedSampler(
            eval_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.eval_batch_size,
            sample_len_list=sample_len_list,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if 0:  # self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer


    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)
        



class HalvaTrainer(Trainer):

    def custom_setup(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        is_encoder_decoder: bool = False,
        loss_alpha: Optional[float] = 0.1,
        disable_dropout: bool = True,
    ):

        if ref_model:
            self.ref_model = ref_model
        
        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.loss_alpha = loss_alpha
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.is_encoder_decoder = is_encoder_decoder
        self.loss_holder=defaultdict(list)
        
        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model is None:
            if hasattr(model, "llama_model"):
                if not hasattr(self.accelerator.unwrap_model(self.model.llama_model), "disable_adapter"):
                    raise ValueError(
                        "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                    )
            elif hasattr(model, "llm_model"):
                if not hasattr(self.accelerator.unwrap_model(self.model.llm_model), "disable_adapter"):
                    raise ValueError(
                        "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                    )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     if self.train_dataset is None or not has_length(self.train_dataset):
    #         return None

    #     if self.args.group_by_modality_length:
    #         lengths = self.train_dataset.modality_lengths
    #         return LengthGroupedSampler(
    #             self.args.train_batch_size,
    #             world_size=self.args.world_size * self.args.gradient_accumulation_steps,
    #             lengths=lengths,
    #             group_by_modality=True,
    #         )
    #     else:
    #         return super()._get_train_sampler()
        
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if 0:  # self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    # FIXME: not tested
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_projector', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(HalvaTrainer, self)._save_checkpoint(model, trial, metrics)

    # FIXME: not tested
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_projector', False):
            pass
        else:
            super(HalvaTrainer, self)._save(output_dir, state_dict)

    def cal_batch_logp(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # decoder only model
        if not self.is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]

        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return per_token_logps

    def accumulate_logps(self, logps, signs):
        unique_signs, indices = torch.unique(signs, sorted=True, return_inverse=True)
        accumulated_logps = torch.zeros(signs.size(0), len(unique_signs) - 1, dtype=logps.dtype, device=logps.device)
        
        for i, sign in enumerate(unique_signs[1:]):
            mask = (signs == sign).float()
            accumulated_logps[:, i] = (logps * mask).sum(dim=-1)
        
        return accumulated_logps

    def concatenated_forward(
        self, model, inputs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        images = inputs["images"]
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        neg_input_ids = inputs["neg_input_ids"]
        neg_labels = inputs["neg_labels"]
        neg_attention_mask = inputs["neg_attention_mask"]
        pos_signs = inputs["pos_signs"]
        neg_signs = inputs["neg_signs"]
           
        max_dim = max(input_ids.shape[1], neg_input_ids.shape[1])
        batch_input_ids = torch.zeros((input_ids.shape[0]*2, max_dim), dtype=input_ids.dtype, device=input_ids.device)
        batch_labels = torch.ones((input_ids.shape[0]*2, max_dim), dtype=labels.dtype, device=labels.device) * -100
        batch_attention_mask = torch.zeros((input_ids.shape[0]*2, max_dim), device=attention_mask.device).to(torch.bool)
        batch_signs = torch.zeros((input_ids.shape[0]*2, max_dim), dtype=pos_signs.dtype, device=pos_signs.device)

        batch_input_ids[:input_ids.shape[0], :input_ids.shape[1]] = input_ids
        batch_input_ids[neg_input_ids.shape[0]:, :neg_input_ids.shape[1]] = neg_input_ids
        batch_labels[:labels.shape[0], :labels.shape[1]] = labels
        batch_labels[neg_labels.shape[0]:, :neg_labels.shape[1]] = neg_labels
        batch_attention_mask[:attention_mask.shape[0], :attention_mask.shape[1]] = attention_mask
        batch_attention_mask[neg_attention_mask.shape[0]:, :neg_attention_mask.shape[1]] = neg_attention_mask
        batch_signs[:pos_signs.shape[0], :pos_signs.shape[1]] = pos_signs
        batch_signs[neg_signs.shape[0]:, :neg_signs.shape[1]] = neg_signs
        
        # calculate logits
        outputs = model(
            input_ids=batch_input_ids,
            images=torch.cat([images, images], dim=0),
            labels=batch_labels,
            attention_mask=batch_attention_mask,
            signs=batch_signs,
        )

        all_logits=outputs.logits.to(torch.float32)
        batch_labels=outputs.labels
        batch_signs=outputs.signs

        all_logps = self.cal_batch_logp(
            all_logits,
            batch_labels,
        )

        if not self.is_encoder_decoder:
            batch_labels = batch_labels[:, 1:].clone()
            batch_signs = batch_signs[:, 1:].clone()
            all_logits = all_logits[:, :-1, :]
        
        len_chosen = input_ids.shape[0]
        pos_logps = all_logps[:len_chosen]
        neg_logps = all_logps[len_chosen:]
                
        return (pos_logps, neg_logps, batch_labels, all_logits, batch_signs)

    def reference_forward(
        self, model, inputs
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        images = inputs["ref_images"]
        input_ids = inputs["ref_input_ids"]
        labels = inputs["ref_labels"]
        attention_mask = inputs["ref_attention_mask"]
        
        outputs=model(
            input_ids=input_ids,
            images=images.squeeze(1),
            attention_mask=attention_mask,
            labels=labels,
        )

        logits = outputs.logits.to(torch.float32)
        labels = outputs.labels

        logps = self.cal_batch_logp(
            logits,
            labels,
        )

        if not self.is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]

        return (logps, labels, logits)


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
               
        # contrastive_loss=0
        # -------------------- cotrastive loss
        (
            pos_logps,
            neg_logps,
            batch_labels,
            all_logits,
            batch_signs,
        ) = self.concatenated_forward(model, inputs)

        loss_mask=batch_labels!=-100
        half_len=pos_logps.shape[0]

        pos_loss_mask = loss_mask[:half_len].float()
        neg_loss_mask = loss_mask[half_len:].float()

        neg_logps = neg_logps*neg_loss_mask # batch x num_tokens
        pos_logps = pos_logps*pos_loss_mask # batch x num_tokens

        # accumulate logps at word level
        batch_signs = batch_signs.masked_fill(batch_signs==-100, 0)
        pos_sign = batch_signs[:half_len, ]
        neg_sign = batch_signs[half_len:, ]

        pos_logps_acc = self.accumulate_logps(pos_logps, pos_sign)
        neg_logps_acc = self.accumulate_logps(neg_logps, neg_sign)

        contrastive_loss = torch.log(1 + torch.exp(neg_logps_acc - pos_logps_acc))
        contrastive_loss = contrastive_loss.mean()

        # -------------------- divergence
        
        (
            logps, labels, logits
        ) = self.reference_forward(model, inputs)

        with torch.no_grad():
            (
                reference_logps, reference_labels, reference_logits
            ) = self.reference_forward(self.ref_model, inputs)
        
        reference_loss_mask=reference_labels!=-100
        vocab_sz = reference_logits.shape[-1]

        reference_logits = F.softmax(reference_logits, dim=-1)
        logits = F.softmax(logits, dim=-1)

        # print(inputs['ref_input_ids'].shape)
        # print(reference_logits.shape, logits.shape)

        divergence = (reference_logits*(reference_logits.log()-logits.log()))
        divergence = divergence * reference_loss_mask.unsqueeze(-1)
        batch_sz = divergence.shape[0]
        divergence = divergence.sum()/batch_sz

        loss = contrastive_loss + self.loss_alpha*divergence

        print(f"[loss: {round(loss.item(), 7)} contrastive_loss:  {round(contrastive_loss.item(), 7)}, divergence: {round(divergence.item(), 7)}]")

        self.loss_holder['contrastive_loss'].append(round(contrastive_loss.item(), 7))
        self.loss_holder['divergence'].append(round(divergence.item(), 7)) 
        self.loss_holder['loss'].append(round(loss.item(), 7)) 

        # print(batch_signs[0])

        return loss
        