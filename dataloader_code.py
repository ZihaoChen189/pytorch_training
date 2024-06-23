import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="./data_10", train=False,
                                        transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

img, target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{} with shuffle was False".format(epoch), imgs, step)
        step += 1

writer.close()
