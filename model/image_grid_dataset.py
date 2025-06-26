import torch
import torchvision as tv
import matplotlib.pyplot as plt
import random
import einops

torch.manual_seed(47)
random.seed(47)

class Combine(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super().__init__()
        # keep your tf exactly the same
        self.tf = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.tk = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 's': 10, 'e': 11 }
        self.ds = tv.datasets.MNIST(root='.', train=train, download=True)
        self.ln = len(self.ds)

    def __len__(self):
        return self.ln

    def __getitem__(self, idx):
        idxs = random.sample(range(self.ln), 4)
        imgs = [self.ds[i][0] for i in idxs]               # PIL
        labels = [self.ds[i][1] for i in idxs]
        tnsrs = [self.tf(img) for img in imgs]             # normalized [−.4 … 2.8]
        stack = torch.stack(tnsrs, dim=0).squeeze()        # (4,28,28)

        # stitch into 56×56
        combo = einops.rearrange(stack, '(h w) ph pw -> (h ph) (w pw)',
                                 h=2, w=2, ph=28, pw=28)
        mean = 0.1307
        std  = 0.3081
        cmb_unnorm = std * combo + mean
        # split into four 14×14 patches
        patch = einops.rearrange(cmb_unnorm, '(h ph) (w pw) -> (h w) ph pw',
                                 ph=14, pw=14)
        labels = [10] + labels + [11]
        
        return combo, patch, torch.tensor(labels)

if __name__ == "__main__":
    ds = Combine()
    cmb, pch, lbl = ds[0]
    print("labels:", lbl)

    # ---- UN-NORMALIZE ----
    # cmb is normalized:   (raw/255 - mean) / std
    # to get back to [0..1] raw, do:
    mean = 0.1307
    std  = 0.3081
    cmb_unnorm = cmb * std + mean       # now roughly background≈0, digits up to 1

    # ---- DISPLAY ----
    # For black background & white digits, just show cmb_unnorm:
    plt.figure(figsize=(4,4))
    plt.imshow(cmb_unnorm.numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

    # If you *really* want to invert (white→black, black→white), do:
    cmb_inv = 1.0 - torch.clamp(cmb_unnorm, 0.0, 1.0)
    plt.figure(figsize=(4,4))
    plt.imshow(cmb_inv.numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

    # ---- PATCH STRIP ----
    # un-normalize the 4 patches and display side by side
    # rearrange (4,14,14) → (14,56)
    strip = einops.rearrange(pch, 'p h w -> h (p w)')
    strip_unnorm = strip * std + mean
    plt.figure(figsize=(6,2))
    plt.imshow(strip_unnorm.numpy(), cmap='gray')
    plt.axis('off')
    plt.show()