import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import datetime
import os
import pickle
import wandb
import tqdm
from patch_and_embed import image_to_patch_columns
from encoder import TransformerEncoder
from transformer import Transformer
from torch.utils.data import random_split, DataLoader, TensorDataset


#TODO: add eval 


def train() -> None:
    """Training function that can be called by wandb agent or directly."""
    # --- setup seed, timestamp, device, W&B ---
    torch.manual_seed(42)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", dev)
    
    # wandb.init() will be called by the agent in sweep mode, or we call it here for single runs
    if not wandb.run:
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project="mlx_wk3_mnist_transformer",
            name=f"mnist_transformer_{ts}",
            config={
                "init_learning_rate": 1e-4,
                "min_learning_rate": 1e-6,
                "batch_size": 1024,
                "num_epochs": 100,
                "num_heads": 8,
                "num_encoders": 8,
                "num_patches": 16,
                "patch_size": 7,
                "stride": 7,
                "dim_patch": 49,
                "dim_proj_V": 25,
                "dim_proj_QK": 100,
                "dim_out": 49,
                "dim_in": 49,
                "mlp_hidden_dim": 25,
                # decoder stuff below
                "max_seq_len": 10,  # Maximum sequence length for decoder input
                "dec_dim_in": 64,
                "dec_dim_out":64,
                "num_decoders": 6,
                "dec_mask_num_heads": 8,
                "dec_cross_num_heads": 32,



            },
        )


# TODO:
# data loaders (images and tokens)
# call model
# set up optimiser
# train loop: for epoch in epochs
    # pass model data (images and tokens)
    # calc loss and accuracy/other metrics
    # optimiser
    # backprop
# save model here or during epochs