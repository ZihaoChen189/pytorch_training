from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logsss")
img = Image.open("data/train/ants_image/0013035.jpg")
print(img)

# To Tensor
trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalise", img_norm, 2)

print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_tensor = trans_to_tensor(img_resize)
writer.add_image("Resize", img_tensor, 3)
print(img_resize)


# Compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_to_tensor])
trans_resize_2 = trans_compose(img)
writer.add_image("Resize-", img_tensor, 1)


# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_to_tensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
