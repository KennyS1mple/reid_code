# CACIOUS CODING
# Data     : 6/29/23  5:26 PM
# File name: test
# Desc     : temp test file for python
import os

from dataset.my_dataset import MyDataset
from opt import opts

opt = opts()
my_dataset = MyDataset(opt)
print(len(my_dataset))

img_path = os.path.join(opt.dataset_path, "obj_img")
print(len(os.listdir(img_path)))
