# CACIOUS CODING
# Data     : 6/1/23  8:51 PM
# File name: inference
# Desc     :
import time

import network.reid_training_model
from opt import opts
from model.model import *
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

transform = transforms.Compose([
    transforms.ToTensor()
])


def pic2tensor(img_path):
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    h, w = img.shape[:2]
    ratio = min(opt.img_size[0] / w, opt.img_size[1] / h)
    new_h, new_w = int(h * ratio), int(w * ratio)
    img = cv2.resize(img, (new_w, new_h))

    grey_bg = np.full((opt.img_size[1], opt.img_size[0], 3), 255 // 2, dtype=np.uint8)
    grey_bg[int(grey_bg.shape[0] / 2 - img.shape[0] / 2):int(grey_bg.shape[0] / 2 + img.shape[0] / 2),
            int(grey_bg.shape[1] / 2 - img.shape[1] / 2):int(grey_bg.shape[1] / 2 + img.shape[1] / 2)] = img

    img = transform(grey_bg)
    return img.to(opt.device).unsqueeze(0)


opt = opts()

opt.device = torch.device('cpu')
model = create_model(opt.arch, opt.heads, opt.head_conv)
model = network.reid_training_model.TrainingModel(model, opt)

model = load_model(model, opt.load_model)
model = model.model
model = model.to(opt.device)
model.eval()

img1_path = "/home/cacious/Pictures/Screenshots/luggage0.png"
img1 = pic2tensor(img1_path)

img2_path = "/home/cacious/Pictures/Screenshots/luggage1.png"
img2 = pic2tensor(img2_path)

img = torch.cat([img1, img2], dim=0)
# print(model(img))
stat = time.time()
output = model(img)[0]['id']
end = time.time()
print(end - stat)
dim = output[..., output.shape[-2] // 2, output.shape[-1] // 2]
dim1 = dim[0]
dim2 = dim[1]
dim1 = nn.functional.normalize(dim1, dim=0)
dim2 = nn.functional.normalize(dim2, dim=0)
cosDistance = torch.matmul(dim1, dim2.T)
print(cosDistance.cpu().data)
