import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./data_10", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output


if __name__ == "__main__":
    loss = nn.CrossEntropyLoss()
    tudui = Tudui()
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        # print(output.shape)
        # print(output)
        # print(targets)
        result_loss = loss(outputs, targets)
        # print(result_loss)
        # result_loss.backward()
        print("ok")
