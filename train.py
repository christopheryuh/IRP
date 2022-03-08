from torchvision import datasets
from torchvision.transforms import ToTensor

import torch

import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")#use GPU if available

parser = argparse.ArgumentParser(description="Experiment Perameters")
parser.add_argument('training_samples', metavar='N', type=int)
parser.add_argument('model_path', metavar='N', type=str)
parser.add_argument('dataset', metavar='N', type=str)

args = parser.parse_args()

training_samples = args.training_samples
model_path = args.model_path
dataset_name = args.dataset


#model = torch.load(model_path)


#load dataset

if dataset_name not in ["CIFAR10","MNIST","FashionMNIST"]: #TODO: Finalize datasets
    raise RuntimeError("Unsupported Dataset") #make sure we have a supported datset

    print(train_data.shape)
#load CIFAR10
if dataset_name == "CIFAR10":
    train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    ) #load train data from torchvision

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    ) #load test data from torchvision

    print("train_data:", train_data,"\n")
    print("test_data:", test_data,"\n")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    print("Train_Loader Iters:",len(train_loader),"\n")
    print("Test_Loader Iters:",len(test_loader),"\n")


#load FashionMNIST
if dataset_name == "FashionMNIST":
    train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    ) #load train data from torchvision

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    ) #load test data from torchvision

    print("train_data:", train_data,"\n")
    print("test_data:", test_data,"\n")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    print("Train_Loader Iters:",len(train_loader),"\n")
    print("Test_Loader Iters:",len(test_loader),"\n")



#load MNIST
if dataset_name == "MNIST":
    train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    ) #load train data from torchvision

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    ) #load test data from torchvision

    print("train_data:", train_data,"\n")
    print("test_data:", test_data,"\n")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    print("Train_Loader Iters:",len(train_loader),"\n")
    print("Test_Loader Iters:",len(test_loader),"\n")