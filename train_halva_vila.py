import os
os.environ["WANDB_PROJECT"] = "HALVA" # for wandb logging

from unittest import mock
from vila.train.train_halva import train
from vila.train.transformer_normalize_monkey_patch import patched_normalize

if __name__ == "__main__":
    with (
        mock.patch('transformers.image_processing_utils.normalize', new=patched_normalize),
        # mock.patch('accelerate.data_loader.BatchSamplerShard.__len__', new=__len__),
        # mock.patch('accelerate.data_loader.BatchSamplerShard.__iter__', new=__iter__)
        ):
            train()