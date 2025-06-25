import pickle
import torch
import torch.nn.functional as F
import numpy as np

# image1 = train_images[0]
# print('shape of image1:', image1[0].shape)
# print('shape of image1 first column:',  image1[0][:, 0].shape)
# print('shape of image1 first row:',  image1[0][0, :].shape)

# # patches = F.unfold(example_image.float(), kernel_size=7, stride=7)
# image = torch.arange(28*28).reshape(1, 28, 28).float()
# image = image.squeeze(0) 
# patches = image.unfold(0, 7, 7).unfold(1, 7, 7)
# columns = patches.contiguous().view(-1, 7*7).t().contiguous().view(-1, 1)

# def image_to_patch_columns(image, patch_size=7, stride=7):
#     """
#     Splits the input image into non-overlapping patches and returns them as columns.
#     Args:
#         image (torch.Tensor): 2D tensor of shape (H, W)
#         patch_size (int): Size of each patch (default: 7)
#         stride (int): Stride for patch extraction (default: 7)
#     Returns:
#         torch.Tensor: Columns of shape (num_patches, patch_size*patch_size)
#     """

#     print('shape of image:', image.shape)
#     patches = image.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
#     print('shape of patches:', patches.shape)
#     columns = patches.contiguous().view(-1, patch_size * patch_size)
#     print('shape of columns:', columns.shape)
#     # print('new: ',columns)
#     return columns

def image_to_patch_columns(images, patch_size=7, stride=7):
    """
    Splits the input images into non-overlapping patches and returns them as columns.
    Args:
        images (torch.Tensor): 3D tensor of shape (B, H, W)
        patch_size (int): Size of each patch (default: 7)
        stride (int): Stride for patch extraction (default: 7)
    Returns:
        torch.Tensor: Columns of shape (B, num_patches, patch_size*patch_size)
    """
    if images.dim() == 2:
        images = images.unsqueeze(0)
    batch_columns = []
    for image in images:
        patches = image.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        columns = patches.contiguous().view(-1, patch_size * patch_size)
        batch_columns.append(columns)
    return torch.stack(batch_columns)

## FOR LOOP IMPLEMENTATION OF PATCHING'
# patches = []
# for i in range(0, 28, 7):
#     for j in range(0, 28, 7):
#         patch = image[i:i+7, j:j+7]
#         patches.append(patch)
# print(patches[0].shape)
# print(patches[0])
# print(patches[1])
# print('number of patches:', len(patches))
# # print(patches[0].view(-1).shape)

# columns = torch.zeros(16, 49)
# for i, patch in enumerate(patches):
#     for i_x, x in enumerate(patch):
#         for i_y, y in enumerate(x):
#             columns[i, i_x*7+i_y] = y

# print(columns.shape)
# print(columns)