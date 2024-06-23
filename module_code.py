import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


test = Tudui()
x = torch.tensor(1.0)
output = test(x)
print(output)
