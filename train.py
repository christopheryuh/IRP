import torch

from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

import tensorflow as tf
from torchsummary import summary

from tqdm import tqdm
from torch.nn import functional as F


import argparse
import numpy as np
from matplotlib import pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")  # use GPU if available

parser = argparse.ArgumentParser(
    description="Experiment Perameters"
)  # get args for the runs
parser.add_argument("training_samples", metavar="N", type=int)
parser.add_argument("batch_size", metavar="N", type=int)
parser.add_argument("model_path", metavar="N", type=str)
parser.add_argument("dataset", metavar="N", type=str)

args = parser.parse_args()

training_samples = args.training_samples
batch_size = args.batch_size
model_path = args.model_path
dataset_name = args.dataset

model = torch.load(model_path).to(device)  # load model
print(model)
# load dataset

if dataset_name not in ["CIFAR10", "MNIST", "FashionMNIST"]:  # get datase
    raise RuntimeError("Unsupported Dataset")  # make sure we have a supported datse

# load CIFAR10
if dataset_name == "CIFAR10":
    cifar10 = tf.keras.datasets.cifar10

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = tf.image.rgb_to_grayscale(train_images, name=None)
    train_images = tf.image.central_crop(train_images, 0.875)
    train_images = np.array(train_images).reshape(-1, 1, 28, 28)

    test_images = tf.image.rgb_to_grayscale(test_images, name=None)
    test_images = tf.image.central_crop(test_images, 0.875)
    test_images = np.array(test_images).reshape(-1, 1, 28, 28)

    train_labels = train_labels.reshape(-1)
    test_labels = test_labels.reshape(-1)

# load FashionMNIST
if dataset_name == "FashionMNIST":
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# load MNIST
if dataset_name == "MNIST":
    mnist = tf.keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


print("Dataset Dowloaded")

train_images = np.array(train_images).reshape(-1, 1, 28, 28)

train_images = train_images / 255.0  # normalize
test_images = test_images / 255.0

ds_list = [
    (torch.tensor(train_images[i]), train_labels[i]) for i in range(len(train_labels))
]  # turn dataset into list with the right corresponding images and labels for shuffling
random.shuffle(ds_list)  # shuffle


train_loader = torch.utils.data.DataLoader(
    ds_list[:training_samples], batch_size=batch_size
)  # get the right number of training samples
test_loader = torch.utils.data.DataLoader(
    [(test_images[i], test_labels[i]) for i in range(len(test_labels))],
    shuffle=True,
    batch_size=batch_size,
)

del train_images, train_labels, test_images, test_labels

print("Train_Loader Iters:", len(train_loader), "\n")
print("Test_Loader Iters:", len(test_loader), "\n")

log_path = (
    f"logs/{dataset_name}/{model_path[-6:-3]}/{training_samples}"  # tensorboard logging
)
print(log_path)
writer = SummaryWriter(log_dir=log_path)


# for b in train_loader:
#     x,y = b
#     print(x.shape)
#     plt.imshow(x.reshape(28,28))
#     print(y)
#     plt.show()
# I used this to make sure the right labels were with the right images


# lr = .1 for linear
# lr = 0.1  for small
lr = 0.1  # for big
optim = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.1)
loss_fn = nn.CrossEntropyLoss()

plateau = 0
min_loss = np.Infinity
patience = 100
done = False

prev_loss = 0

losslist = []
steps = 0

print("Learning Rate:", lr)
for e in tqdm(range(1000)):
    e_steps = 1
    e_losslist = []
    # training and validation
    for batch in train_loader:
        x, y = batch
        optim.zero_grad()
        y = y.to(device)
        x = x.float().to(device)

        loss = loss_fn(model(x), y.long())
        # print(loss)
        loss.backward()
        optim.step()

        steps += 1
        e_losslist.append(loss)

        e_steps += 1

    losslist.append((sum(e_losslist) / e_steps).detach())  # avg loss over epoch
    writer.add_scalar("Train_Loss", sum(e_losslist) / e_steps, e)

    if min_loss < sum(e_losslist) / e_steps:  # end training at convergance
        plateau += 1
        # print(plateau, min_loss, sum(e_losslist) / e_steps)
    else:
        plateau = 0

    min_loss = min(sum(e_losslist) / e_steps, min_loss)

    if plateau >= patience:
        done = True

    with torch.no_grad():  # get validation accuracy and validation loss
        v_loss = []
        v_steps = 0
        accuracylist = []
        for batch in test_loader:
            x, y = batch
            x = x.to(device).reshape((-1, 1, 28, 28))
            y = y.to(device)

            y_hat = model(x.float())
            loss = loss_fn(y_hat, y.long())

            v_steps += 1
            v_loss.append(loss)
            accuracy = (y_hat.argmax(axis=1) == y).sum() / (y.shape[0])
            accuracylist.append(accuracy)

        final_acc = (sum(accuracylist)) / len(accuracylist)
        final_loss = sum(v_loss) / v_steps

        writer.add_scalar("Test_Accuracy", final_acc, e)
        writer.add_scalar("Test_loss", final_loss, e)
    if done:
        break

# TODO: save to pickle
