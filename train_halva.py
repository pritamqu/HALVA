import os
os.environ["WANDB_PROJECT"] = "HALVA" # for wandb logging

from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
from llava.train.train_halva import train


if __name__ == "__main__":
    train()