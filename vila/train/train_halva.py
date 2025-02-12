import copy
import os
import logging
from typing import Dict, Optional, Sequence, List
import warnings

import torch
import transformers
import random
import json
from PIL import Image

from transformers import HfArgumentParser, AutoTokenizer, AutoConfig, LlamaForCausalLM
from transformers.modeling_utils import unwrap_model
from transformers import set_seed

from torch.utils.data import Dataset
from vila.train.llava_trainer import LLaVATrainer
from vila.train.halva_trainer import HalvaTrainer
from vila.train.args import TrainingArguments, ModelArguments, DataArguments
from vila.train.callbacks.autoresume_callback import AutoResumeCallback

from vila import conversation as conversation_lib
from vila.model import *
from vila.train.utils import (
    get_checkpoint_path,
    prepare_config_for_training,
    vision_resolution_elevation,
    unit_test_rope_scaling,
    mprint,
)
from vila.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                             IMAGE_TOKEN_INDEX)
from vila.mm_utils import is_gemma_tokenizer, tokenizer_image_token, opencv_extract_frames, process_image
from transformers import TrainerCallback
from peft.peft_model import PeftModelForCausalLM
MASK_PLACEHOLDER_START= "<MASK>"
MASK_PLACEHOLDER_END= "</MASK>"


local_rank = None



from dataclasses import dataclass, field
import transformers
from typing import Dict, Optional, Sequence, List

def moving_average(data, window_size=10):
    smoothed_data = []
    for i in range(len(data)):
        window = data[max(0, i - window_size + 1):i + 1]
        smoothed_data.append(sum(window) / len(window))
    return smoothed_data


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    ref_data_path: str = field(default=None,
                           metadata={"help": "Path to the reference data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    data_mixture: str = "llava_1_5_mm_align"
    eval_data_mixture: str = None
    vflan_no_system_prompt: bool = False
    downsample_video: bool = False

    # for video training
    num_video_frames: int = 8



@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    mm_projector: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_resolution: Optional[int] = field(default=-1)
    interpolate_mode: Optional[str] = field(default="linear")
    drop_path_rate: Optional[float] = field(default=0.)
    s2: bool = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")
    s2_max_split_size: int = field(default=336)
    loss_alpha: Optional[float] = field(default=0.) # regulate influence of KL regularizer



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    tune_vision_tower: bool = field(default=False)
    tune_language_model: bool = field(default=False)
    tune_mm_projector: bool = field(default=False)
    model_dtype: str = field(default="torch.bfloat16")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    total_time_limit: int = field(
        default=-1, metadata={"help": "Timeout limit for this job (in minutes)."}
    )
    pre_terminate_time: int = field(
        default=10,
        metadata={
            "help": "Time to terminate the task inadvance (minutes), saveing checkpoints needs time."
        },
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def setup_model(model_args, data_args, training_args):

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    resume_from_checkpoint = False
    if "mpt" in model_args.model_name_or_path or \
        "mistral" in model_args.model_name_or_path.lower() or \
            "mixtral" in model_args.model_name_or_path.lower() or \
                "gemma" in model_args.model_name_or_path.lower():
        raise NotImplementedError(model_args.model_name_or_path)
    
    else:
        ## llm and default multimodal model
        model_cls = LlavaLlamaModel
        config = LlavaLlamaConfig.from_pretrained(
            model_args.model_name_or_path,
            resume=resume_from_checkpoint
        )

    if getattr(config, "resume_path", None) is not None:
        config.resume_path = model_args.model_name_or_path

    ## extra configurations
    prepare_config_for_training(config, model_args, training_args, data_args)

    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args,
    )



    vision_resolution_elevation(model, config)
    # This is an empty func.
    # It would be overwritten by unit test script.
    if unit_test_rope_scaling(model, model.llm.config, training_args):
        return

    # Take a look on model architecture.
    mprint(model)

    model.llm.config.use_cache = False
    ## set tunnable parameters
    logging.warning(
        "You are setting tunable parameters for the model. Previous args include 'freeze_backbone' and 'tune_mm_mlp_adapter' are deprecated.\n Notice: default value of tune_xxx is False, which means you would not tune this part."
    )
    model.get_llm().requires_grad_(training_args.tune_language_model)
    mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")
    if model.get_vision_tower():
        model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
        model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
        mprint(f"vision tower {training_args.tune_vision_tower}")
        mprint(f"mm projector {training_args.tune_mm_projector}")
    if not any([training_args.tune_language_model, training_args.tune_vision_tower, training_args.tune_mm_projector]):
        logging.warning(
            "You are not tuning any part of the model. Please check if this is intended."
        )

    def need_to_modify_do_sample(generation_config):
        if generation_config.do_sample is False:
            if (
                generation_config.temperature is not None
                and generation_config.temperature != 1.0
            ):
                return True
            if generation_config.top_p is not None and generation_config.top_p != 1.0:
                return True
        return False

    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    ## quantize training @yunhao: be careful here
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.llm.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model.llm = prepare_model_for_kbit_training(
            model.llm, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # adding lora; check if adding lora does not make any change in require grad for the other part of the model
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        mprint("Adding LoRA adapters...")
        # adding lora to llm only
        # model.llm.model = get_peft_model(model.llm.model, lora_config)
        peft_model = get_peft_model(model.llm, lora_config)
        model.llm = peft_model # .get_base_model()
        model.llm.model.embed_tokens = peft_model.get_base_model().model.embed_tokens # it may throw error w/o this


    # @yunhao: tokenizer instantiation is moved into build_llm
    tokenizer = model.tokenizer
    # @yunhao: may move this block into method "build_llm"
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.llm,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.llm,
            )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    # kentang-mit@: It will be useful in on-the-fly packing
    model.llm.pad_token_id = tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = tokenizer.model_max_length
    # if training_args.lora_enable:
    #     model.base_model.model.llm.pad_token_id = tokenizer.pad_token_id

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.num_video_frames = data_args.num_video_frames
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    
    ## TODO pay attention to quantize
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)


    return model, tokenizer





def split_string_by_mask_and_tokenize(string, tokenizer):
    tokens = []
    signs = []
    start_tag = MASK_PLACEHOLDER_START
    end_tag = MASK_PLACEHOLDER_END
    start_index = 0
    mask_tag=1
    while True:
        start_pos = string.find(start_tag, start_index)

        if start_pos == -1:
            skip_ss=2
            tokens.extend(tokenizer(string[start_index:]).input_ids[skip_ss:-1]) # non_masked_parts
            signs.extend([0]*(len(tokens)-len(signs)))
            # print("start_pos -1 ")
            break
        end_pos = string.find(end_tag, start_pos + len(start_tag))
        # if end_pos == -1:
        #     skip_ss=2
        #     tokens.extend(tokenizer(string[start_index:]).input_ids[skip_ss:-1]) # non_masked_parts
        #     signs.extend([0]*(len(tokens)-len(signs)))
        #     # print("end_pos -1 ")
        #     break

        if start_index==0:
            skip_ss=1
        else:
            skip_ss=2

        tokens.extend(tokenizer(string[start_index:start_pos]).input_ids[skip_ss:-1]) # non_masked_parts
        signs.extend([0]*(len(tokens)-len(signs))) # adding 0s for new unmasked tokens
        
        # --------- masked part
        # if there is a puntuation right after masked token, add the puctuation and then encode
        # because tokenization scheme differes if punctuation is at the beg or end
        # FIXME: this is not a full-proof setup, works well with . or ,; need to test more
        # punctuation=r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        if string[end_pos + len(end_tag):end_pos + len(end_tag) + 1] in r""".,""":
            skip_ss=2
            punct = string[end_pos + len(end_tag):end_pos + len(end_tag) + 1]
            with_punct=string[start_pos + len(start_tag):end_pos]+punct
            with_punct=with_punct.replace(' .', '. ').replace(' ,', ', ') # FIXME: add other cases as well
            # print(with_punct)
            tokens.extend(tokenizer(with_punct).input_ids[skip_ss:-1]) # masked_parts

            # adding 1's for new masked tokens; do not count punc in the sign
            # FIXME: assuming adding punc increased the length by 1; 
            # this is true for llama tokenizer but may need modification for a different tokenizer

            signs.extend([mask_tag]*(len(tokens)-len(signs)-1))
            signs.extend([0]*1) 
            start_index = end_pos + len(end_tag) + 1 # moved 1 pos
            mask_tag+=1 # each masks are marked with a diff tracking number

        elif string[end_pos + len(end_tag):end_pos + len(end_tag) + 2] == "'s":
            skip_ss=2
            punct = string[end_pos + len(end_tag):end_pos + len(end_tag) + 2]
            with_punct=string[start_pos + len(start_tag):end_pos]+punct
            with_punct=with_punct.replace(" 's", "'s ")
            tokens.extend(tokenizer(with_punct).input_ids[skip_ss:-1]) # masked_parts

            # adding 1s for new masked tokens; do not count punc in the sign
            # FIXME: assuming adding punc increased the length by 1; this is true for llama tokenizer
            signs.extend([mask_tag]*(len(tokens)-len(signs)-1))
            signs.extend([0]*1) 
            start_index = end_pos + len(end_tag) + 2 # moved 2 pos
            mask_tag+=1 # each masks are marked with diff number

        else:
            skip_ss=2
            tokens.extend(tokenizer(string[start_pos + len(start_tag):end_pos]).input_ids[skip_ss:-1]) # masked_parts
            signs.extend([mask_tag]*(len(tokens)-len(signs))) # adding 1s for new masked tokens
            start_index = end_pos + len(end_tag)
            mask_tag+=1 # each masks are marked with diff number


    return tokens, signs

def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids

def tokenizer_image_token_masked(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):

    prompt = prompt.split('<image>')
    assert len(prompt)==2, 'assuming the only users give image, and it is a single turn conversation'

    pre_image_prompt, post_image_prompt = prompt

    input_ids = []
    # pre_image_prompt should not have MASK tag based on llava prompt template; # TODO: make this better
    assert MASK_PLACEHOLDER_START not in pre_image_prompt
    input_ids.extend(tokenizer(pre_image_prompt).input_ids)
    input_ids.extend([image_token_index])
    signs = [0]*len(input_ids)

    post_image_prompt_tokens, signs_maksed = split_string_by_mask_and_tokenize(post_image_prompt, tokenizer)
    
    signs.extend(signs_maksed)
    input_ids.extend(post_image_prompt_tokens)

    input_ids.extend([tokenizer.eos_token_id])
    signs.extend([0])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(signs, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids, signs


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    
    assert has_image, "this code may not be ready to handle non image setup"
    conv = conversation_lib.default_conversation.copy()
    if no_system_prompt:
        conv.system = ""
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    ############ just for sanity check; to make sure our tokenization with MASK tag is working correctly
    ref_sources = copy.deepcopy(sources)
    assert ref_sources[0][2]['from']=='gpt-ref'
    ref_sources[0][2]['from']='gpt'
    sources = [sources[0][:-1]]
    ref_sources = [ref_sources[0][0], ref_sources[0][2]] # creating a ref copy with unmasked responses
    ref_sources=[ref_sources]
    
    ref_conversations = []
    for i, source in enumerate(ref_sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        ref_conversations.append(conv.get_prompt())

    # print(ref_conversations) # TODO: remove
    ref_input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in ref_conversations], dim=0)

    ############ end


    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())


    # print(conversations) # TODO: remove

    # Tokenize conversations
    assert len(conversations)==1
    input_ids, signs = tokenizer_image_token_masked(conversations[0], tokenizer, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0)
    signs=signs.unsqueeze(0)

    # sanity check; if there is some problem in tokenization with MASK tag -> discard
    if (input_ids!=ref_input_ids).sum()>0:
        print('conversations: ', conversations)
        print('ref_conversations', ref_conversations)
        print(f'[Error in tokenization] input_ids: {input_ids}, ref_input_ids: {ref_input_ids}')
        return None

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(ref_conversations, targets): # we are using here ref_conversations as conversations has MASK tags
        # conversation=conversations[0]
        # target=targets[0]

        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                if i > 0 and not is_gemma_tokenizer(tokenizer):
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                if i > 0 and not is_gemma_tokenizer(tokenizer):
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}"
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        signs=signs,
    )


def preprocess_v1_ref(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    if no_system_prompt:
        conv.system = ""
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # print(conversations)
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                if i > 0 and not is_gemma_tokenizer(tokenizer):
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                if i > 0 and not is_gemma_tokenizer(tokenizer):
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}"
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )
    
def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        concat_values = "".join([sentence["value"] for sentence in source])
        for sid, sentence in enumerate(source):
            # In multimodal conversations, we automatically prepend '<image>' at the start of the first sentence if it doesn't already contain one.
            if sid == 0 and DEFAULT_IMAGE_TOKEN not in concat_values:
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" + sentence["value"]
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence_chunks = [chunk.strip() for chunk in sentence["value"].split(DEFAULT_IMAGE_TOKEN)]
                sentence_chunks = [
                    chunk + " " if not (chunk.endswith("\n")) else chunk for chunk in sentence_chunks[:-1]
                ] + [sentence_chunks[-1]]
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n".join(sentence_chunks).strip()

                replace_token = DEFAULT_IMAGE_TOKEN
                if "mmtag" in conversation_lib.default_conversation.version:
                    replace_token = "<Image>" + replace_token + "</Image>"
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

class HallDataset(Dataset):
    """Dataset for Hal style training."""

    def __init__(self, data_path: str, 
                 ref_data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(HallDataset, self).__init__()

        self.data_args = data_args
        list_data_dict, neg_list_data_dict = self.prepare_data_dict(data_path)

        if ref_data_path in [None, "none"]:
            mprint(f"we will use {data_path} as reference")
            ref_data_dict=None
        else:
            mprint(f"we will use {ref_data_path} as reference")
            ref_data_dict = self.get_ref_data_dict(ref_data_path, len(list_data_dict))
            assert len(list_data_dict)==len(neg_list_data_dict)==len(ref_data_dict)

        mprint("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.neg_list_data_dict = neg_list_data_dict
        self.ref_data_dict = ref_data_dict

        if self.data_args.image_folder=='default':
            self.IMAGE_DIRS={
                'textvqa': '/h/pritam/pritam_ssd004/datasets/textvqa', 
                'gqa': '/h/pritam/pritam_ssd004/datasets/gqa', 
                'vg': '/h/pritam/pritam_ssd004/datasets/vg/images', 
                'coco': '/scratch/ssd004/datasets/MSCOCO2017', 
                'ocr_vqa': '/h/pritam/pritam_ssd004/datasets/ocr_vqa',
            }
        else:
            self.IMAGE_DIRS={
            'textvqa': self.data_args.image_folder+'textvqa', 
            'gqa': self.data_args.image_folder+'gqa', 
            'vg': self.data_args.image_folder+'vg', 
            'coco': self.data_args.image_folder+'coco', 
            'ocr_vqa': self.data_args.image_folder+'ocr_vqa',
            }

        # # sanity
        # self.check_if_all_images_are_avbl()
            
    def get_ref_data_dict(self, data_path, num_samples):
        data=json.load(open(data_path))
        assert len(data)>num_samples
        return data[:num_samples]
        
    def get_image_file_path(self, image_file):
        _src = image_file.split('/')[0]
        _rest = '/'.join(image_file.split('/')[1:])
        image_file = os.path.join(self.IMAGE_DIRS[_src], _rest)
        return image_file

    def check_if_all_images_are_avbl(self):
        flag=True
        for sample in self.list_data_dict:
            if 'image' in sample:
                image_file = sample['image']
                image_file = self.get_image_file_path(image_file)
                if not os.path.isfile(image_file):
                    print('[ERROR:] missing images: {image_file}')
                    flag=False

        print('finished checking image paths')
        if not flag:
            raise FileNotFoundError()

    def prepare_data_dict(self, data_path):
        pos_data=[]
        neg_data=[]
        
        data=json.load(open(data_path))
        closed_subset=[sample for sample in data if sample['tag'] == 'closed']
        mprint(f'number of adv samples {len(closed_subset)}')
        open_subset=[sample for sample in data if sample['tag'] == 'open']
        mprint(f'number of rand samples {len(open_subset)}')
        one_word=[sample for sample in data if sample['tag']=='qa']

        if True:
            mprint('getting equal number of yes-no samples')
            random.seed(42)
            random.shuffle(one_word)
            one_word_pos=[k for k in one_word if k['raw_answer'].lower()=='yes']
            one_word_neg=[k for k in one_word if k['raw_answer'].lower()=='no']
            one_word_min=min(len(one_word_pos), len(one_word_neg))
            one_word_eq=[]
            one_word_eq.extend(one_word_pos[:one_word_min])
            one_word_eq.extend(one_word_neg[:one_word_min])
            one_word=one_word_eq

        mprint(f'number of qa samples {len(one_word)}')

        data=[]
        data.extend(closed_subset)
        data.extend(open_subset)
        data.extend(one_word)
        random.seed(42)
        random.shuffle(data)

        mprint(f'current data size {len(data)}')

        for sample in data:

            conversation=[]
            conversation.append({'from': 'human', 
                                'value': sample['question'],
                                })
            conversation.append({'from': 'gpt', 
                                'value': sample['correct_answer_masked']})
            conversation.append({'from': 'gpt-ref', # for debug purpose
                                'value': sample['correct_answer']})
            
            pos_data.append({'conversations': conversation,
                            'id': sample['id'],
                            'image': sample['image'],
                            })
            
            neg_conversation=[]
            neg_conversation.append({'from': 'human', 
                                'value': sample['question'],
                                })
            neg_conversation.append({'from': 'gpt', 
                                'value': sample['hallucinated_answer_masked']})
            neg_conversation.append({'from': 'gpt-ref', # for debug purpose
                                'value': sample['hallucinated_answer']})
            
            neg_data.append({'conversations': neg_conversation,
                            'id': sample['id'],
                            'image': sample['image'],
                            })
            
        return pos_data, neg_data

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        neg_sources = self.neg_list_data_dict[i]
        assert sources['id']==neg_sources['id']
        if isinstance(i, int):
            sources = [sources]
            neg_sources = [neg_sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # same images are used for pos and neg
        
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                image = torch.stack(
                    [process_image(
                                   # img, 
                                   self.get_image_file_path(img),
                                   self.data_args, self.data_args.image_folder) for img in image_file]
                )
            else:
                image_file = self.get_image_file_path(image_file)
                image = process_image(image_file, self.data_args, self.data_args.image_folder)

        
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            neg_sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in neg_sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            neg_sources = copy.deepcopy([e["conversations"] for e in neg_sources])

        # print('*'*10, 'Positives\n', sources, '*'*10, 'Negatives\n', neg_sources)
        
        data_dict = preprocess_v1(sources, self.tokenizer, 
                                  has_image=('image' in self.list_data_dict[i]))
        
        neg_data_dict = preprocess_v1(neg_sources, self.tokenizer, 
                                      has_image=('image' in self.neg_list_data_dict[i]))

        if isinstance(i, int):
            final_data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0],
                                neg_input_ids=neg_data_dict["input_ids"][0],
                                neg_labels=neg_data_dict["labels"][0],
                                pos_signs=data_dict["signs"][0],
                                neg_signs=neg_data_dict["signs"][0],
                                )
                
        # image exist in the data
        if "image" in self.list_data_dict[i]:
            # final_data_dict["image"] = image
            if len(image.shape) == 4:
                final_data_dict["image"] = image
            else:
                final_data_dict["image"] = image.unsqueeze(0)
        else:
            crop_size = self.data_args.image_processor.crop_size
            # final_data_dict["image"] = None
            final_data_dict["image"] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])

        # add reference samples
        if self.ref_data_dict is not None:
            ref_final_data_dict = self.ref_getitem(i) 
            final_data_dict['ref_input_ids'] = ref_final_data_dict["input_ids"]
            final_data_dict['ref_labels'] = ref_final_data_dict["labels"]
            final_data_dict['ref_image'] = ref_final_data_dict["image"]
        else:
            final_data_dict['ref_input_ids'] = final_data_dict["input_ids"]
            final_data_dict['ref_labels'] = final_data_dict["labels"]
            final_data_dict['ref_image'] = final_data_dict["image"]


        return final_data_dict


    def ref_getitem(self, i) -> Dict[str, torch.Tensor]:
        sources = self.ref_data_dict[i]

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
  
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                image = torch.stack(
                    [process_image(
                                   # img,
                                   self.get_image_file_path(img),
                                   self.data_args, self.data_args.image_folder) for img in image_file]
                )
            else:
                image_file = self.get_image_file_path(image_file)
                image = process_image(image_file, self.data_args, self.data_args.image_folder)

            
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        

        data_dict = preprocess_v1_ref(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.ref_data_dict[i]
            ),
        )
        if isinstance(i, int):
            final_data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])


        # image exist in the data
        if "image" in self.ref_data_dict[i]:
            # final_data_dict["image"] = image
            if len(image.shape) == 4:
                final_data_dict["image"] = image
            else:
                final_data_dict["image"] = image.unsqueeze(0)
        else:
            crop_size = self.data_args.image_processor.crop_size
            # final_data_dict["image"] = None
            final_data_dict["image"] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])

        return final_data_dict


@dataclass
class DataCollatorForHallDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels, neg_input_ids, neg_labels, pos_signs, neg_signs, ref_input_ids, ref_labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "neg_input_ids", "neg_labels", "pos_signs", "neg_signs", "ref_input_ids", "ref_labels"))
        

        input_ids = torch.nn.utils.rnn.pad_sequence(
                                                input_ids,
                                                batch_first=True,
                                                padding_value=self.tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(
                                                labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
        

        neg_input_ids = torch.nn.utils.rnn.pad_sequence(
                                                neg_input_ids,
                                                batch_first=True,
                                                padding_value=self.tokenizer.pad_token_id)
        
        neg_labels = torch.nn.utils.rnn.pad_sequence(
                                                neg_labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)

        pos_signs = torch.nn.utils.rnn.pad_sequence(
                                                pos_signs,
                                                batch_first=True,
                                                padding_value=0)

        neg_signs = torch.nn.utils.rnn.pad_sequence(
                                                neg_signs,
                                                batch_first=True,
                                                padding_value=0)
        

        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
                                                ref_input_ids,
                                                batch_first=True,
                                                padding_value=self.tokenizer.pad_token_id)
        
        ref_labels = torch.nn.utils.rnn.pad_sequence(
                                                ref_labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
        

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        neg_input_ids = neg_input_ids[:, :self.tokenizer.model_max_length]
        neg_labels = neg_labels[:, :self.tokenizer.model_max_length]

        pos_signs = pos_signs[:, :self.tokenizer.model_max_length]
        neg_signs = neg_signs[:, :self.tokenizer.model_max_length]

        ref_input_ids = ref_input_ids[:, :self.tokenizer.model_max_length]
        ref_labels = ref_labels[:, :self.tokenizer.model_max_length]        

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            neg_input_ids=neg_input_ids,
            neg_labels=neg_labels,
            neg_attention_mask=neg_input_ids.ne(self.tokenizer.pad_token_id),
            pos_signs=pos_signs,
            neg_signs=neg_signs,
            ref_input_ids=ref_input_ids,
            ref_labels=ref_labels,
            ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'ref_image' in instances[0]:
            images = [instance['ref_image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['ref_images'] = torch.stack(images)
            else:
                batch['ref_images'] = images

        # print(batch['images'].shape)
     
        return batch



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset for contrastive tuning."""

    train_dataset = HallDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                ref_data_path=data_args.ref_data_path,
                                data_args=data_args)
    data_collator = DataCollatorForHallDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
        

class SaverCallback(TrainerCallback):
    
    "A callback that prints a message at the end of training"
    def on_train_end(self, args, state, control, **kwargs):
        # save model
        if isinstance(kwargs['model'], PeftModelForCausalLM):
            torch.cuda.synchronize()
            state_dict = get_peft_state_maybe_zero_3(
                kwargs['model'].named_parameters(), "none"
            )
            kwargs['model'].save_pretrained(args.output_dir)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                kwargs['model'].named_parameters()
            )
            kwargs['model'].config.save_pretrained(args.output_dir)
            kwargs['model'].save_pretrained(args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(args.output_dir, 'non_lora_trainables.bin'))
    

def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.run_name = training_args.output_dir.split("/")[-1]
    local_rank = training_args.local_rank
    
    assert model_args.version in ['v1'], "This code supports llama3 conversation template."
    # make a copy to load ref model; safe coding
    ref_model_args=copy.deepcopy(model_args)
    ref_data_args=copy.deepcopy(data_args)
    ref_training_args=copy.deepcopy(training_args)

    set_seed(training_args.seed)

    # pack model setup in a func and call twice for model and ref_model
    mprint('Loading online model')
    model, tokenizer = setup_model(model_args, data_args, training_args)

    # using the base model as the reference model
    mprint(f'Loading reference model: {ref_model_args.model_name_or_path}')
    ref_training_args.lora_enable = False # reference model should not have lora
    ref_model, _ = setup_model(ref_model_args, ref_data_args, ref_training_args)
    
    # freeze reference model
    for n,p in ref_model.named_parameters():
        p.requires_grad = False
    
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        # training_args=training_args,
    )

    trainer = HalvaTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        # callbacks=callbacks, 
        **data_module
    )
    trainer.custom_setup(
                    model=model,
                    ref_model=ref_model,
                    label_pad_token_id=IGNORE_INDEX,
                    padding_value=tokenizer.pad_token_id,
                    loss_alpha=model_args.loss_alpha,
                    )
    
    # trainer.add_callback(SaverCallback()) # FIXME: i should not need this

    print(
        "length of dataloader:",
        len(trainer.get_train_dataloader()),
        len(trainer.train_dataset),
        flush=True,
    )
    print(
        "[GPU memory] before trainer",
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        flush=True,
    )

    resume_from_checkpoint=False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    model.llm.config.use_cache = True

    if training_args.lora_enable: # halva
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            # model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            model.llm.save_pretrained(training_args.output_dir, state_dict=state_dict) # lora
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()
