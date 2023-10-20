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
    
#Create Model instance and set to a variable

model = Model()
mse = nn.MSELoss() #Loss generator. This is how we get our Loss. Loss compares output that we received, based on what was calculated, to our desired output. What's the difference? Prediction vs result and how close are they together? Let's us calculate the prediction accuracy of the model. Means^2Error, Loss

optimizer = optim.SGD(model.parameters(), lr=0.01) # Going to update our parameters. Stotastic Gradient Descent (SGD). Optimizes parameters to get them to the lowest possible point on bell curve. lr is learning rate.

#Sample inputs

#Makes a 5 X 1 matrix
inputs = torch.rand((5, 1), dtype=torch.float32) #float32 occupies 32 bits
factor = 2  #multiplying all the numbers in matrix by 2
targets = inputs * factor

#Training Loop 
#epoch = training iteration

for epoch in range(10000):
    #Forward pass
    outputs = model.forward(inputs)
    loss = mse(outputs, targets)

    #Backward pass and optimization
    optimizer.zero_grad() #zero-gradient accumulation. By default, pytorch will accumulate gradients. This is a best practice. Making sure gradient is only from current iteration, not from ones before. There are some instances where you would want to accumulate gradients for more complex models.
    loss.backward()
    optimizer.step() #able to update our model parameters; refined

    if epoch % 100 == 0: # every 100 iterations, print the training iteration # and the loss
        print(f'epoch {epoch} loss: {loss:.4f}')

#to run in terminal, $ python main.py
