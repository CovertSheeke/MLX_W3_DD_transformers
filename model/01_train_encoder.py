import torch
import encoder
import patch_and_embed
import pickle
import wandb 
import tqdm
import datetime
import os

## setup timestamp and seed for reproducibility
torch.manual_seed(42)
ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

## Load the MNIST dataset

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist_trainset.pkl')
with open(os.path.abspath(data_path), 'rb') as f:
    trainset = pickle.load(f)

## Set up the device for training
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", dev)

## initialize WandB for logging
wandb.init(
    entity=os.environ.get("WANDB_ENTITY"),
    project="mlx_wk3_mnist_transformer",
    name=f"mnist_transformer_{ts}",
    config={
        "learning_rate": 0.0001,
        "batch_size": 1024,
        "num_epochs": 100,
        "num_heads": 8,
        "num_encoders": 8,
        "num_patches": 16,  # Assuming a fixed number of patches, e.g., 16 for MNIST
        "patch_size": 7,
        "stride": 7,
        "dim_patch": 49,  # Assuming each patch is flattened to a vector of size 49 (7x7)
        "dim_proj_V": 49,  # Projection dimension for value matrix, can be same as dim_in
        "dim_proj_QK": 49,  # Projection dimension for key&query matrices, can be same as dim_in
        "dim_out": 49,
        "dim_in": 49,  # Output dimension, must be same as dim_in
        "mlp_hidden_dim": 25,  # Hidden dimension for MLP
    }
)

## Initialize the encoder
enc = encoder.TransformerEncoder(wandb.config).to(dev)

# Prepare the dataset and dataloader
print("Preparing dataset and dataloader...")
# Scale dataset to be from [0,1], move to device
ds = trainset.data.float().div(255.0).to(dev)
# Standardise to zero mean/unit variance (MNIST stats)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
ds = ds.sub_(MNIST_MEAN).div_(MNIST_STD)
# Move targets to device as well
data_targets = trainset.targets.to(dev)
###TODO: how are the targets shaped? do we need to onehot encode?
dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(ds, data_targets),
    batch_size=wandb.config.batch_size,
    shuffle=True
)
opt = torch.optim.Adam(enc.parameters(), lr=wandb.config.learning_rate)

# Define the training loop
print("Starting training...")
for epoch in range(wandb.config.num_epochs):
    prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{wandb.config.num_epochs}")
    for idx, (imgs, trgs) in enumerate(prgs):
        imgs, trgs = imgs.to(dev), trgs.to(dev)
        # print("imgs shape: ", imgs.shape, "trgs shape: ", trgs.shape)
        img_embs = patch_and_embed.image_to_patch_columns(imgs, patch_size=wandb.config.patch_size, stride=wandb.config.stride)
        img_embs = img_embs.to(dev)
        # print(f"Image embeddings shape: {img_embs.shape}, Targets shape: {trgs.shape}")
        opt.zero_grad()
        loss, accuracy = enc(img_embs, trgs)
        loss.backward()
        opt.step()
        wandb.log({"loss": loss.item(), "accuracy": accuracy, "epoch": epoch + 1, "batch": idx + 1})
        # print("MADE IT HERE")
        if idx % 100 == 0:
            prgs.set_postfix({"loss": loss.item()})
            print(f"Epoch {epoch + 1}, Batch {idx + 1}, Loss: {loss.item()}")


wandb.finish()
print("Training complete. Model saved and logged to WandB.")