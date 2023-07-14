# CACIOUS CODING
# Data     : 5/29/23  9:24 PM
# File name: get_obj_img
# Desc     : get object of interest

import os
import cv2
import sys


def main():
    img_path = sys.argv[1] + "/images/train"  # 图片路径
    label_path = sys.argv[1] + "/labels/train"  # txt文件路径
    save_path = sys.argv[1] + "/obj_img"  # 保存路径
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_total = []
    label_total = []
    imgfile = os.listdir(img_path)
    labelfile = os.listdir(label_path)

    for filename in imgfile:
        name, type = os.path.splitext(filename)
        if type == ('.jpg' or '.png'):
            img_total.append(name)
    for filename in labelfile:
        name, type = os.path.splitext(filename)
        if type == '.txt':
            label_total.append(name)

    for _img in img_total:
        if _img in label_total:
            filename_img = _img + '.jpg'
            path = os.path.join(img_path, filename_img)
            img = cv2.imread(path)  # 读取图片，结果为三维数组
            filename_label = _img + '.txt'
            w = img.shape[1]  # 图片宽度(像素)
            h = img.shape[0]  # 图片高度(像素)
            # 打开文件，编码格式'utf-8','r+'读写
            with open(os.path.join(label_path, filename_label), "r+", encoding='utf-8', errors="ignor") as f:
                for line in f:
                    msg = line.split(" ")  # 根据空格切割字符串，最后得到的是一个list
                    x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                    y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                    x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                    y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                    print(filename_img)
                    img_roi = img[y1:y2, x1:x2]  # 剪裁，roi:region of interest
                    cv2.imwrite(os.path.join(save_path, filename_img), img_roi)
        else:
            continue


if __name__ == '__main__':
    main()
