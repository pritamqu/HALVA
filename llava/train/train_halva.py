import os
import copy
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import random
import torch
import glob
import shutil
import transformers
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DESCRIPTION_SEPARATOR
from torch.utils.data import Dataset
from llava.train.halva_trainer import HalvaTrainer
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from string import punctuation
from PIL import Image
from transformers import TrainerCallback
from peft.peft_model import PeftModelForCausalLM
from dataclasses import dataclass, field

MASK_PLACEHOLDER_START= "<MASK>"
MASK_PLACEHOLDER_END= "</MASK>"
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def moving_average(data, window_size=10):
    smoothed_data = []
    for i in range(len(data)):
        window = data[max(0, i - window_size + 1):i + 1]
        smoothed_data.append(sum(window) / len(window))
    return smoothed_data


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    loss_alpha: Optional[float] = field(default=0.) # regulate influence of KL regularizer

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    ref_data_path: str = field(default=None,
                           metadata={"help": "Path to the reference data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
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
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
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
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources





def split_string_by_mask_and_tokenize(string, tokenizer):

    # TODO: this is a bit hacky solution
    # works fine with the current data and llama2 tokenizer
    # we figured a relatively better solution by assigning a new token 
    # to MASK; will release that in the final verion.
        
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

        if start_index==0:
            skip_ss=1
        else:
            skip_ss=2

        tokens.extend(tokenizer(string[start_index:start_pos]).input_ids[skip_ss:-1]) # non_masked_parts
        signs.extend([0]*(len(tokens)-len(signs))) # adding 0s for new unmasked tokens
        
        # --------- masked part
        if string[end_pos + len(end_tag):end_pos + len(end_tag) + 1] in r""".,""":
            skip_ss=2
            punct = string[end_pos + len(end_tag):end_pos + len(end_tag) + 1]
            with_punct=string[start_pos + len(start_tag):end_pos]+punct
            with_punct=with_punct.replace(' .', '. ').replace(' ,', ', ') # add other punctuation as well
            # print(with_punct)
            tokens.extend(tokenizer(with_punct).input_ids[skip_ss:-1]) # masked_parts

            # adding 1's for new masked tokens; do not count punc in the sign
            # assuming adding punc increased the length by 1; 
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
            # assuming adding punc increased the length by 1; this is true for llama tokenizer
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


def tokenizer_image_token_masked(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):

    prompt = prompt.split('<image>')
    assert len(prompt)==2, 'assuming the only users give image, and it is a single turn conversation'

    pre_image_prompt, post_image_prompt = prompt

    input_ids = []
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
    has_image: bool = False
) -> Dict:
    
    assert has_image, "this code may not be ready to handle non image setup"
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # start of sanity check
    # to make sure our tokenization with MASK tag is working correctly
    # this part can be commented out
    
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

    ref_input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in ref_conversations], dim=0)

    # end of sanity check


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
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
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
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
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
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

    

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
            rank0_print(f"we will use {data_path} as reference")
            ref_data_dict=None
        else:
            rank0_print(f"we will use {ref_data_path} as reference")
            ref_data_dict = self.get_ref_data_dict(ref_data_path, len(list_data_dict))
            assert len(list_data_dict)==len(neg_list_data_dict)==len(ref_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.neg_list_data_dict = neg_list_data_dict
        self.ref_data_dict = ref_data_dict

        if self.data_args.image_folder=='default':
            self.IMAGE_DIRS={
                'textvqa': '/h/anonymous/anonymous_ssd004/datasets/textvqa', 
                'gqa': '/h/anonymous/anonymous_ssd004/datasets/gqa', 
                'vg': '/h/anonymous/anonymous_ssd004/datasets/vg/images', 
                'coco': '/scratch/ssd004/datasets/MSCOCO2017', 
                'ocr_vqa': '/h/anonymous/anonymous_ssd004/datasets/ocr_vqa',
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
        rank0_print(f'number of adv samples {len(closed_subset)}')
        open_subset=[sample for sample in data if sample['tag'] == 'open']
        rank0_print(f'number of rand samples {len(open_subset)}')
        one_word=[sample for sample in data if sample['tag']=='qa']

        if True:
            rank0_print('getting equal number of yes-no samples')
            random.seed(42)
            random.shuffle(one_word)
            one_word_pos=[k for k in one_word if k['raw_answer'].lower()=='yes']
            one_word_neg=[k for k in one_word if k['raw_answer'].lower()=='no']
            one_word_min=min(len(one_word_pos), len(one_word_neg))
            one_word_eq=[]
            one_word_eq.extend(one_word_pos[:one_word_min])
            one_word_eq.extend(one_word_neg[:one_word_min])
            one_word=one_word_eq

        rank0_print(f'number of qa samples {len(one_word)}')

        data=[]
        data.extend(closed_subset)
        data.extend(open_subset)
        data.extend(one_word)
        random.seed(42)
        random.shuffle(data)

        rank0_print(f'current data size {len(data)}')

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
        if 'image' in sources[0]: 
            processor = self.data_args.image_processor
            image_file = self.list_data_dict[i]['image']
            image_file = self.get_image_file_path(image_file)
            image = Image.open(image_file).convert('RGB')

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
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
        if 'image' in self.list_data_dict[i]:
            final_data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            final_data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

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
        if 'image' in sources[0]: 
            processor = self.data_args.image_processor
            image_file = self.ref_data_dict[i]['image']
            image_file = self.get_image_file_path(image_file)
            image = Image.open(image_file).convert('RGB')

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess_v1_ref(
            sources,
            self.tokenizer,
            has_image=('image' in self.ref_data_dict[i]), 
            )
        
        if isinstance(i, int):
            final_data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0],
                                )
        # image exist in the data
        if 'image' in self.ref_data_dict[i]:
            final_data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            final_data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return final_data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, 
                 instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch
    
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
    
def setup_llava(model_args, data_args, training_args):

    
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    rank0_print(f"compute_dtype: {compute_dtype}")

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
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
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
        
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
      
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    return model, tokenizer


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    assert model_args.version in ['v1', 'vicuna_v1'], "This code supports v1 and vicuna_v1 conversation template."

    # make a copy to load ref model; safe coding
    ref_model_args=copy.deepcopy(model_args)
    ref_data_args=copy.deepcopy(data_args)
    ref_training_args=copy.deepcopy(training_args)

    rank0_print('Loading online model')
    model, tokenizer = setup_llava(model_args, data_args, training_args)

    # using the base model as the reference model
    rank0_print(f'Loading reference model: {ref_model_args.model_name_or_path}')
    ref_training_args.lora_enable = False # reference model should not have lora
    ref_model, _ = setup_llava(ref_model_args, ref_data_args, ref_training_args)
    
    # freeze reference model
    for n,p in ref_model.named_parameters():
        p.requires_grad = False
    

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_args)
    
    trainer = HalvaTrainer(model=model,
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    loss_alpha=model_args.loss_alpha,
                    args=training_args,
                    label_pad_token_id=IGNORE_INDEX,
                    padding_value=tokenizer.pad_token_id,
                    **data_module)

    trainer.add_callback(SaverCallback()) # i should not need this

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        






if __name__ == "__main__":
    train()
