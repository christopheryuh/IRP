import torch

from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

import tensorflow as tf
from torchsummary import summary

import argparse
import numpy as np
from matplotlib import pyplot as plt
import random

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")#use GPU if available

parser = argparse.ArgumentParser(description="Experiment Perameters")
parser.add_argument('training_samples', metavar='N', type=int)
parser.add_argument('batch_size', metavar='N', type=int)
parser.add_argument('model_path', metavar='N', type=str)
parser.add_argument('dataset', metavar='N', type=str)

args = parser.parse_args()

training_samples = args.training_samples
batch_size = args.batch_size
model_path = args.model_path
dataset_name = args.dataset

model = torch.load(model_path).to(device)


#load dataset

if dataset_name not in ["CIFAR10","MNIST","FashionMNIST"]: #TODO: Finalize datasets
    raise RuntimeError("Unsupported Dataset") #make sure we have a supported datset

#load CIFAR10
if dataset_name == "CIFAR10":
    cifar10 = tf.keras.datasets.cifar10

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = tf.image.rgb_to_grayscale(train_images, name=None)
    train_images = tf.image.central_crop(train_images, .875)


#load FashionMNIST
if dataset_name == "FashionMNIST":
    cifar10 = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#load MNIST
if dataset_name == "MNIST":
    mnist = tf.keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


print("Dataset Dowloaded")

train_images = np.array(train_images).reshape(-1,1,28,28)

train_images = train_images / 255.0
test_images = test_images / 255.0

ds_list = [(train_images[i], train_labels[i]) for i in range(len(train_labels))]
print(ds_list[0])
random.shuffle(ds_list)
print(ds_list[0])

train_loader = torch.utils.data.DataLoader(ds_list[:training_samples], batch_size=batch_size)
test_loader = torch.utils.data.DataLoader([(test_images[i],test_labels[i]) for i in range(len(test_labels))],shuffle=True,batch_size=batch_size)


print("Train_Loader Iters:",len(train_loader),"\n")
print("Test_Loader Iters:",len(test_loader),"\n")

summary(model, (1,28,28,1))


for b in train_loader:
    x,y = b
    print(x.shape)
    plt.imshow(x.reshape(28,28))
    print(y)
    plt.show()



optim = torch.optim.SGD(model.parameters(), lr=.01, momentum=0.9)
scheduler = ReduceLROnPlateau(optim, 'min')
loss_fn = nn.CrossEntropyLoss()

plateau = 0
min_loss = 0
patience = 100
done = False

losslist = []

#TODO: ADD VALIDATON

for e in range(500):

    e_steps = 1 
    e_losslist = []
    #training and validation
    for batch in train_loader:
        x,y = batch
        x.to(device)
        y.to(device)
        optim.zero_grad()

        y_hat = model(x)
        loss = loss_fn(y_hat,y)

        #TODO: Tensorboard




        e_losslist.append(loss)

        e_steps += 1 

        if min_loss < loss:
            plateau += 1
        else:
            plateau = 0

        min_loss = min(loss,min_loss)

        if plateau >= patience:
            done = True
    losslist.append(sum(e_losslist)/e_steps)#avg loss over epoch
    
    if done:
        break


#TODO: save to a .csv