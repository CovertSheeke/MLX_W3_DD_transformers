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

## Initialize the encoder
enc = encoder.TransformerEncoder().to(dev)

## initialize WandB for logging
wandb.init(
    project="mlx_wk3_mnist_transformer",
    entity="ethangledwards",  # Replace with your WandB entity
    name=f"mnist_transformer_{ts}",
    config={
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_epochs": 2,
        "patch_size": 7,
        "stride": 7
    }
)

# Prepare the dataset and dataloader
print("Preparing dataset and dataloader...")
ds = trainset.data.float().to(dev)
data_targets = trainset.targets.to(dev)
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
        img_embs = patch_and_embed.image_to_patch_columns(imgs, patch_size=wandb.config.patch_size, stride=wandb.config.stride)
        img_embs = img_embs.to(dev)
        # print(f"Image embeddings shape: {img_embs.shape}, Targets shape: {trgs.shape}")
        opt.zero_grad()
        loss = enc(img_embs, trgs)
        loss.backward()
        opt.step()
        wandb.log({"loss": loss.item(), "epoch": epoch + 1, "batch": idx + 1})
        # print("MADE IT HERE")
        if idx % 100 == 0:
            prgs.set_postfix({"loss": loss.item()})
            print(f"Epoch {epoch + 1}, Batch {idx + 1}, Loss: {loss.item()}")
        

wandb.finish()
print("Training complete. Model saved and logged to WandB.")