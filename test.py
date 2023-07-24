# CACIOUS CODING
# Data     : 6/29/23  5:26 PM
# File name: test
# Desc     : temp test file for python
import os

import PIL.Image
import numpy

from dataset.my_dataset import MyDataset
from opt import opts

# opt = opts()
# my_dataset = MyDataset(opt)
# print(len(my_dataset))
#
# img_path = os.path.join(opt.dataset_path, "obj_img")
# print(len(os.listdir(img_path)))

img = PIL.Image.open("dataset/luggage1.png").convert("RGB")
img.show()
img1 = numpy.asarray(img)
print(img1.shape)
