# CACIOUS CODING
# Data     : 6/20/23  1:37 PM
# File name: extract_img
# Desc     : 根据label文件名提取帧

import os
import shutil
import sys


img_dir = sys.argv[1]
label_dir = sys.argv[2]
target_dir = sys.argv[3]

label_names = os.listdir(label_dir)
for base_name in label_names:
    base_name = os.path.splitext(base_name)[0]
    shutil.move(os.path.join(img_dir, base_name + ".jpg"),
                os.path.join(target_dir, base_name + ".jpg"))
    print(base_name)
