# CACIOUS CODING
# Data     : 6/1/24  8:01 PM
# File name: test
# Desc     :
import time

import network.reid_training_model
from opt import opts
from model.model import *
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

transform = transforms.Compose([
    transforms.ToTensor()
])


opt = opts()

opt.device = torch.device('cpu')
model = create_model(opt.arch, opt.heads, opt.head_conv)
model = network.reid_training_model.TrainingModel(model, opt)
model = load_model(model, opt.load_model)
model = model.to(opt.device)
model.eval()
# print(model.state_dict)

img_path = "/media/cacious/share/luggage_detect/reid_dataset/labeled/reid_dataset/obj_img/dawei003902.jpg"
img = np.asarray(Image.open(img_path), dtype=np.uint8)
h, w = img.shape[:2]
ratio = min(opt.img_size[0] / w, opt.img_size[1] / h)
new_h, new_w = int(h * ratio), int(w * ratio)
img = cv2.resize(img, (new_w, new_h))

grey_bg = np.full((opt.img_size[1], opt.img_size[0], 3), 255 // 2, dtype=np.uint8)
grey_bg[int(grey_bg.shape[0] / 2 - img.shape[0] / 2):int(grey_bg.shape[0] / 2 + img.shape[0] / 2),
                int(grey_bg.shape[1] / 2 - img.shape[1] / 2):int(grey_bg.shape[1] / 2 + img.shape[1] / 2)] = img
img = cv2.cvtColor(grey_bg, cv2.COLOR_RGB2BGR)
cv2.imshow("1", img)
cv2.waitKey(0)

img = transform(grey_bg)
img = img.to(opt.device).unsqueeze(0)

start = time.time()
output = model(img)
end = time.time()
print(end - start)
print(output)
print(torch.argmax(output, dim=1))
