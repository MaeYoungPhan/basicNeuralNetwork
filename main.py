import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module): # Model class derives from nn.Module. Is used to keep track of all of the parameters that we have. Parameters mean the weights and biases of the model. Typically, we have an input (x), weight (w), and bias (b). Formula is usually (y) is going to be equal to x*w + b. 

    def __init__(self): # constructor. Initializes parameters
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, input): #This method is our forward pass. When called, will go through the neural network and go from an input to an output.
        out = self.linear(input) # this layer is performing the y = xw + b. Linear transformation
        return out