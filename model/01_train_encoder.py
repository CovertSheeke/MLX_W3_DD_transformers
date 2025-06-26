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
from torch.utils.data import random_split, DataLoader, TensorDataset

# Globals
VAL_SIZE = 10000  # Fixes the number sizes of samples for validation split

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """evaluate:Run model on dataloader and return (avg_loss, avg_accuracy)."""
    model.eval()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0
    with torch.no_grad():# tells PyTorch “don’t keep track of gradients
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            img_embs = image_to_patch_columns(
                imgs,
                patch_size=wandb.config.patch_size,
                stride=wandb.config.stride,
            ).to(device)
            loss, acc = model(img_embs, targets)
            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc

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
            },
        )

    # --- load and normalise MNIST data ---
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist_trainset.pkl")
    with open(os.path.abspath(data_path), "rb") as f:
        fullset = pickle.load(f)
    # x = raw_pixel/255.0 => brings values into [0,1]
    # x = (x - 0.1307) / 0.3081 => centers around 0 with unit variance
    ds = fullset.data.float().div(255.0)# Turns the original uint8 pixel values (0 up to 255) into floats between 0 and 1.
    ds = ds.sub_(0.1307).div_(0.3081)  # MNIST normalisation
    targets = fullset.targets

    # --- split into train/val ---
    train_size = len(ds) - VAL_SIZE
    train_ds, val_ds = random_split(
        TensorDataset(ds, targets),
        [train_size, VAL_SIZE],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=wandb.config.batch_size, shuffle=False)

    # --- model, optimiser ---
    model = TransformerEncoder(wandb.config).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config.init_learning_rate)
    # Set up a scheduler to linearly decay the learning rate from initial to a minimum value
    min_lr = wandb.config.min_learning_rate  # You can adjust this minimum learning rate as needed
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser,
        start_factor=1.0,
        end_factor=min_lr / wandb.config.init_learning_rate,
        total_iters=wandb.config.num_epochs,
    )

    # Set up ckpt saving folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- training & validation loop ---
    best_accuracy = 0.0
    best_ckpt_filename = ""
    for epoch in range(1, wandb.config.num_epochs + 1):
        model.train()
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{wandb.config.num_epochs}")
        for batch_idx, (imgs, trgs) in enumerate(loop, start=1):
            imgs, trgs = imgs.to(dev), trgs.to(dev)
            img_embs = image_to_patch_columns(
                imgs,
                patch_size=wandb.config.patch_size,
                stride=wandb.config.stride,
            ).to(dev)
            optimiser.zero_grad()
            loss, acc = model(img_embs, trgs)
            # Save ckpt if acc is better than previous best
            if acc > best_accuracy:
                # Delete previous best ckpt if it exists
                if best_ckpt_filename:
                    os.remove(os.path.join(CHECKPOINT_DIR, best_ckpt_filename))
                best_ckpt_filename = f"mnist_transformer_best_{epoch}_{ts}.pth"
                best_accuracy = acc
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        CHECKPOINT_DIR,
                        best_ckpt_filename
                    )
                )
            # Continue into backprop
            loss.backward() # Compute gradients
            optimiser.step() # Update parameters
            wandb.log({"train_loss": loss.item(), "train_acc": acc, "epoch": epoch})
            if batch_idx % 100 == 0: # Every 100 batches update the tqdm bar.
                loop.set_postfix(loss=loss.item())
        
        # end of epoch → Evaluate on the validation set
        val_loss, val_acc = evaluate(model, val_loader, dev)
        
        # Step the scheduler to update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        wandb.log({"val_loss": val_loss, "val_acc": val_acc, "learning_rate": current_lr, "epoch": epoch})
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}")
    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            CHECKPOINT_DIR,
            f"mnist_transformer_final_epoch_{ts}.pth"
        )
    )

    wandb.finish()
    print("Training complete.")

def run_sweep():
    """Run a wandb sweep to test different num_heads and num_encoders."""
    sweep_config = {
        'method': 'grid',# Grid sweep: tries every combination of num_heads x num_encoders
        'name': 'mnist_transformer_sweep',
        'metric': {
            'goal': 'maximize',
            'name': 'val_acc'
        },
        'parameters': {
            'num_heads': {
                'values': [4, 8, 16]
            },
            'num_encoders': {
                'values': [4, 6, 8, 12]
            },
            # Fixed parameters
            'init_learning_rate': {'value': 1e-4},
            'min_learning_rate': {'value': 1e-6},
            'batch_size': {'value': 1024},
            'num_epochs': {'value': 30},  # Reduced for faster sweep
            'num_patches': {'value': 16},
            'patch_size': {'value': 7},
            'stride': {'value': 7},
            'dim_patch': {'value': 49},
            'dim_proj_V': {'value': 25},
            'dim_proj_QK': {'value': 100},
            'dim_out': {'value': 49},
            'dim_in': {'value': 49},
            'mlp_hidden_dim': {'value': 25},
        },
        'early_terminate': {
            'type': 'hyperband',# Stops bad run early.
            'min_iter': 5,
            'max_iter': 30,
            's': 2,
            'eta': 3
        }
    }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project="mlx_wk3_mnist_transformer",
        entity=os.environ.get("WANDB_ENTITY")
    )
    
    print(f"Starting sweep with ID: {sweep_id}")
    print("You can also run additional agents with:")
    print(f"wandb agent {os.environ.get('WANDB_ENTITY', 'your-entity')}/mlx_wk3_mnist_transformer/{sweep_id}")
    
    # Run the sweep agent
    wandb.agent(sweep_id, train, count=12)  # Launches 3 head-options x 4 encoder-options = 12 combinations for grid search

def main():
    """Main function to choose between single run or sweep."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        run_sweep()
    else:
        train()

if __name__ == "__main__":
    main()
