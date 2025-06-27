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


# Wandb sweep configuration
SWEEP_CONFIG = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'train_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'init_learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-4
        },
        'batch_size': {
            'values': [256, 512, 1024]
        },
        'num_heads': {
            'values': [12, 16, 20]
        },
        'num_encoders': {
            'values': [6, 8, 10]
        },
        'num_decoders': {
            'values': [6, 8, 10]
        },
        'dim_proj_V': {
            'values': [16, 25, 32, 49, 64]
        },
        'dim_proj_QK': {
            'values': [100, 128]
        },
        'mlp_hidden_dim': {
            'values': [16, 64]
        },
        'dec_mask_num_heads': {
            'values': [4, 8, 16]
        },
        'dec_cross_num_heads': {
            'values': [8, 12, 16]
        }
    }
}

#TODO: add eval 
TOKEN2IDX = {
  "0": 0,
  "1": 1,
  "2": 2,
  "3": 3,
  "4": 4,
  "5": 5,
  "6": 6,
  "7": 7,
  "8": 8,
  "9": 9,
  "<pad>": 10,
  "<start>": 11,
  "<stop>": 12,
}

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
            name=f"mnist_transformer_enc_and_dec_{ts}",
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
                "dec_dim_in": 49,
                "dec_dim_out":49,
                "num_decoders": 6,
                "dec_mask_num_heads": 8,
                "dec_cross_num_heads": 8,
            },
        )

    # load dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist_trainset.pkl")
    with open(os.path.abspath(data_path), "rb") as f:
        fullset = pickle.load(f)
    ds = fullset.data.float().div(255.0)
    ds = ds.sub_(0.1307).div_(0.3081)  # MNIST normalisation
    targets = fullset.targets

    train_ds = TensorDataset(ds, targets)

    train_loader = DataLoader(
        train_ds,
        batch_size=wandb.config.batch_size,
        shuffle=True)
    

    
    # model, optimiser, cross entropy loss, and scheduler
    model = Transformer(wandb.config).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config.init_learning_rate)
    min_lr = wandb.config.min_learning_rate  # You can adjust this minimum learning rate as needed
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser,
        start_factor=1.0,
        end_factor=min_lr / wandb.config.init_learning_rate,
        total_iters=wandb.config.num_epochs,
    )
    cel = nn.CrossEntropyLoss()

    # Set up ckpt saving folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CHECKPOINT_DIR = os.path.join(project_root, "checkpoints", f"enc_and_dec")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- training loop ---
    for epoch in range(wandb.config.num_epochs):
        model.train()
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.num_epochs},", leave=False)
        for images, labels in loop:
            imgs, lbls = images.to(dev), labels.to(dev)
            start_token = torch.full((lbls.size(0), 1), TOKEN2IDX["<start>"], device=dev)
            # print('start token', start_token.shape)
            # print('lbls', lbls.shape)
            # Add dimension to lbls to make it [batch_size, 1] for concatenation
            lbls_expanded = lbls.unsqueeze(1)  # Convert from [1024] to [1024, 1]
            lbls = torch.cat([start_token, lbls_expanded], dim=1)  # Add start token to labels
            # print('labels', labels)
            img_embs = image_to_patch_columns(
                imgs,
                patch_size=wandb.config.patch_size,
                stride=wandb.config.stride,
            ).to(dev)

            optimiser.zero_grad()
            output = model(img_embs, lbls)
            
            # Create target labels by shifting left and adding stop token
            # labels[:, 1:] removes the start token from each sequence (keeping original lbls)
            # Then we add the stop token at the end of each sequence
            batch_size = lbls.size(0)
            stop_token = torch.full((batch_size, 1), TOKEN2IDX["<stop>"], device=dev)
            targets = torch.cat([lbls[:, 1:], stop_token], dim=1)  # Use labels (with start token) instead of lbls
            
            # Adjust logits to match target sequence length
            logits = output[:, :targets.size(1), :]  # Take only as many predictions as we have targets
            logits = logits.reshape(-1, logits.size(-1))  # Reshape for loss calculation
            targets = targets.reshape(-1)  # Reshape for loss calculation
            # print("input labels (with start token):", lbls[0])
            # print("targets (shifted with stop token):", targets[0])
            loss = cel(logits, targets)  # Calculate cross-entropy loss
            # acc = ### calculate accuracy here
            loss.backward()
            optimiser.step()
            wandb.log({
                "train_loss": loss.item(),
                # "train_acc": acc,
                "epoch": epoch + 1,
                "learning_rate": optimiser.param_groups[0]['lr'],
            })
            loop.set_postfix(loss=loss.item())#, accuracy=acc)
        scheduler.step()

class image_to_token_dataset(torch.utils.data.Dataset):
    """Dataset that converts images to token sequences."""
    def __init__(self, images, tokens):
        self.images = images
        self.tokens = tokens

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.tokens[idx]

def create_sweep_config_files():
    """Create sweep configuration files for easy experimentation."""
    import yaml
    
    # Bayes optimization sweep config
    bayes_config = {
        'method': 'bayes',
        'metric': {
            'name': 'train_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'init_learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'batch_size': {
                'values': [256, 512, 1024, 2048]
            },
            'num_heads': {
                'values': [4, 8, 16]
            },
            'num_encoders': {
                'values': [4, 6, 8, 12]
            },
            'num_decoders': {
                'values': [4, 6, 8, 12]
            },
            'dim_proj_V': {
                'values': [16, 25, 32, 49, 64]
            },
            'dim_proj_QK': {
                'values': [64, 100, 128, 256]
            }
        }
    }
    
    # Random search sweep config
    random_config = {
        'method': 'random',
        'metric': {
            'name': 'train_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'init_learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'batch_size': {
                'values': [256, 512, 1024, 2048]
            },
            'num_heads': {
                'values': [4, 8, 16]
            },
            'num_encoders': {
                'values': [4, 6, 8, 12]
            },
            'num_decoders': {
                'values': [4, 6, 8, 12]
            }
        }
    }
    
    # Write config files
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    with open(os.path.join(project_root, "sweep_config.yaml"), 'w') as f:
        yaml.dump(bayes_config, f, default_flow_style=False)
    
    with open(os.path.join(project_root, "sweep_config_random.yaml"), 'w') as f:
        yaml.dump(random_config, f, default_flow_style=False)
    
    print("Created sweep configuration files:")
    print("- sweep_config.yaml (Bayes optimization)")
    print("- sweep_config_random.yaml (Random search)")

def run_sweep():
    """Run wandb sweep using the SWEEP_CONFIG defined in this script."""
    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG,
        project="mlx_wk3_mnist_transformer",
        entity=os.environ.get("WANDB_ENTITY")
    )
    
    print(f"Starting wandb sweep with ID: {sweep_id}")
    print(f"View sweep at: https://wandb.ai/{os.environ.get('WANDB_ENTITY', 'your-entity')}/mlx_wk3_mnist_transformer/sweeps/{sweep_id}")
    
    # Run the sweep with 10 runs by default
    wandb.agent(sweep_id, train, count=50)

if __name__ == "__main__":
    import sys
    
    # Check if --sweep argument is passed
    if "--sweep" in sys.argv:
        run_sweep()
    else:
        # Run single training
        train()






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