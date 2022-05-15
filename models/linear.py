import torch

from torch.nn import Sequential, Linear, ReLU, Softmax,Flatten
from torchsummary import summary

use_cuda = torch.cuda.is_available()

# model = Sequential(
#     *[
#     Flatten(),
#     Linear(28*28,14*28),
#     ReLU(),
#     Linear(14*28,7*14),
#     ReLU(),
#     Linear(14*7,49),
#     ReLU(),
#     Linear(49,10),
#     Softmax()
#     ]
# ).cuda()




model = Sequential(
    *[
    Flatten(),
    Linear(28*28,14*28),
    ReLU(),
    Linear(14*28,7*14),
    ReLU(),
    Linear(14*7,49),
    ReLU(),
    Linear(49,49),
    ReLU(),
    Linear(49,49),
    ReLU(),
    Linear(49,20),
    ReLU(), 
    Linear(20,20),
    ReLU(), 
    Linear(20,10),
    Softmax()
    ]
).cuda()

#for debuging

# model = Sequential(

#     *[
#         Flatten(),
#         Linear(28*28,10),
#         Softmax()
#     ]
# )


print(model)
summary(model,(1,28,28,1))

torch.save(model,"models/lin.pt")