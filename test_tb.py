from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/bees_image/16838648_415acd9e3f.jpg"
# image_path = "data/train/ants_image/0013035.jpg"

img = Image.open(image_path)
img_array = np.array(img)
# print(img_array.shape)
writer.add_image("test image", img_array, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()
