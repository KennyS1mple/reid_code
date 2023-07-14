# CACIOUS CODING
# Data     : 6/1/23  8:03 PM
# File name: rename
# Desc     :
import os


pic_dir = "../dataset/luggage/obj_img"
for index, file_name in enumerate(os.listdir(pic_dir)):
    os.rename(os.path.join(pic_dir, file_name), os.path.join(pic_dir, str(index) + ".jpg"))
