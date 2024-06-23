import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./data_10", train=True, download=False, transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="./data_10", train=False, download=False, transform=dataset_transform)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img, target)
# print(test_set.classes[target])
# img.show()

print(test_set[0])

writer = SummaryWriter("log-1")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
