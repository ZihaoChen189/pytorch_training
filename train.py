import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

device = torch.device("mps")

train_data = torchvision.datasets.CIFAR10(root="./data_10", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="./data_10", train=False, transform=torchvision.transforms.ToTensor())
print("The length of training data is {}.".format(len(train_data)))
print("The length of test data is {}.".format(len(test_data)))

train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)

tudui = Tudui()
tudui = tudui.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)


total_train_step = 0
total_test_step = 0
epoch = 2

writer = SummaryWriter("log_pro")

for i in range(epoch):
    print("EPOCH: {}".format(i+1))

    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("Step: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("The whole test loss: {}".format(total_test_loss))
    print("The whole test accuracy: {}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/len(test_data), total_test_step)
    total_test_step += 1

    torch.save(tudui, "Already Save: tudui_epoch_{}.pth".format(i+1))

writer.close()
