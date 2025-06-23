import pickle
import torch
import torch.nn.functional as F
import numpy as np

with open('./data/mnist_trainset.pkl', 'rb') as f:
    train_images = pickle.load(f)

image1 = train_images[0]
print('shape of image1:', image1[0].shape)
print('shape of image1 first column:',  image1[0][:, 0].shape)
print('shape of image1 first row:',  image1[0][0, :].shape)

# patches = F.unfold(example_image.float(), kernel_size=7, stride=7)
image = torch.arange(28*28).reshape(1, 28, 28).float()
image = image.squeeze(0) 
patches = image.unfold(0, 7, 7).unfold(1, 7, 7)
columns = patches.contiguous().view(-1, 7*7).t().contiguous().view(-1, 1)


patches = []
for i in range(0, 28, 7):
    for j in range(0, 28, 7):
        patch = image[i:i+7, j:j+7]
        patches.append(patch)
print(patches[0].shape)
print(patches[0])
print(patches[1])
print('number of patches:', len(patches))
# print(patches[0].view(-1).shape)

columns = torch.zeros(16, 49)
for i, patch in enumerate(patches):
    for i_x, x in enumerate(patch):
        for i_y, y in enumerate(x):
            columns[i, i_x*7+i_y] = y

print(columns.shape)
print(columns)