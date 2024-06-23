import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./data_10", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, 64, drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


if __name__ == "__main__":
    tudui = Tudui()
    for data in dataloader:
        imgs, targets = data
        print(imgs.shape)
        # output = torch.reshape(imgs, (1, 1, 1, -1))
        output = torch.flatten(imgs)
        print(output.shape)

        output = tudui(output)
        print(output.shape)
