from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open("data/train/ants_image/0013035.jpg")
# print(img)

writer = SummaryWriter("logss")
tensor_train = transforms.ToTensor()
tensor_img = tensor_train(img)
# print(tensor_img)

writer.add_image("Tensor_img", tensor_img)
writer.close()
