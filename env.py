### HARDWARE ###

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# fix: GPU OOM (TF exhausts GPU memory, crashing PyTorch)
# not work now
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

### GLOBAL VARS ###

import torch

path = os.path.dirname(os.path.abspath(__file__))
n_gpu = torch.cuda.device_count()

import warnings
import pandas

warnings.filterwarnings(
    action="ignore",
    message="You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset",
    category=UserWarning
)
warnings.simplefilter(
    action="ignore",
    category=pandas.errors.PerformanceWarning
)

### LIBRARY VARS ###

import datasets
datasets.disable_progress_bar()
