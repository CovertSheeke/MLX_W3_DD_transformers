import pickle
import torchvision
import torchvision.transforms as transforms

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST('data', download=True, train=True, transform=transform)
testset = torchvision.datasets.MNIST('data', download=True, train=False, transform=transform)
with open('./data/mnist_trainset.pkl', 'wb') as f:
    pickle.dump(trainset, f)
with open('./data/mnist_testset.pkl', 'wb') as f:
    pickle.dump(testset, f)
print("Loaded datasets")