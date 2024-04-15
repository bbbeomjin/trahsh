
import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.activation = nn.SiLU()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,stride=1,padding=2)
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.conv3 = nn.Conv2d(6, 16, kernel_size=5)
        self.MaxPool_3 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.conv4 = nn.Conv2d(16, 120, kernel_size=5)
        self.FC_1 = nn.Linear(120,84)
        self.FC_2 = nn.Linear(84,10)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.MaxPool_1(self.activation(self.conv1(x)))

        x_1 = self.MaxPool_2(self.activation(self.conv2(x)))
        x_2 = self.MaxPool_3(self.activation(self.conv3(x)))
        x = x_1+x_2
        x = self.activation(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.FC_1(x))
        x = self.softmax(self.FC_2(x))
        return x


# class CustomMLP(nn.Module):
#     """ Your custom MLP model

#         - Note that the number of model parameters should be about the same
#           with LeNet-5
#     """

#     def __init__(self):

#         # write your codes here

#     def forward(self, img):

#         # write your codes here

#         return output
