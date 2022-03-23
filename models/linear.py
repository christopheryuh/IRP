import torch

from torch.nn import Sequential, Linear, ReLU, Softmax,Flatten
from torchsummary import summary

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")#use GPU if available

model = Sequential(
    *[
    Flatten(),
    Linear(28*28,14*28),
    ReLU(),
    Linear(14*28,14*28),
    ReLU(),
    Linear(14*28,14*14),
    ReLU(),
    Linear(14*28,14*14),
    ReLU(),
    Linear(14*14,7*7),
    ReLU(),
    Linear(14*14,7*7),
    ReLU(),
    Linear(7*7,49),
    ReLU(),
    Linear(49,20),
    ReLU(),
    Linear(20,10),
    ReLU(),
    Linear(10,10),
    Softmax()
    ]
).to(device)

print(model)
summary(model,(1,28,28,1))

torch.save(model,"models/linear.pt")