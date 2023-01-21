### HARDWARE ###

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# fix: GPU OOM (TF exhausts GPU memory, crashing PyTorch)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

### DATASETS ###

import sys
sys.path.append("/home/turx/EvalBase")
import evalbase
for ds_name in evalbase.datasets:
    evalbase.datasets[ds_name]["approaches"] = ["new"]
from evalbase import newsroom, realsumm, summeval, tac2010

### GLOBAL VARS ###

import torch

path = os.path.dirname(os.path.abspath(__file__))
n_gpu = torch.cuda.device_count()

### LIBRARY VARS ###

import datasets
datasets.disable_progress_bar()

### METRICS ###

corr_metrics = ["pearsonr", "kendalltau", "spearmanr"]
