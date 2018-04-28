# -*- coding: utf-8 -*-
"""
@Time    : 2018/3/23 17:57
@Author  : Elvis
"""
"""
 dataUtils.py
  
"""
import os
import shutil
import numpy as np

# dataset_dir = "/home/elvis/data/attribute/ai_challenger_zsl2018_train_test_a_20180321"
dataset_dir = "/home/elvis/data/attribute/ai_challenger_zsl2018_test_b_20180423"
superclass = "vehicles"

# train_dir = "zsl_a_fruits_train_20180321"
# train_dir = "zsl_a_%s_train_20180321" % superclass
train_dir = "zsl_b_%s_train_20180321" % superclass
# train_img_dir = "zsl_a_fruits_train_images_20180321"
# train_img_dir = "zsl_a_%s_train_images_20180321" % superclass
train_img_dir = "zsl_b_%s_train_images_20180321" % superclass

attr_path = os.path.join(dataset_dir, train_dir)
train_img_path = os.path.join(dataset_dir, train_dir, train_img_dir)

# label_name_path = os.path.join(attr_path,
#                                "zsl_a_%s_train_annotations_label_list_20180321.txt" % superclass)
label_name_path = os.path.join(attr_path,
                               "zsl_b_%s_train_annotations_label_list_20180321.txt" % superclass)

name2label = dict()
train_labels = []
with open(label_name_path, "r") as fr:
    for line in fr.readlines():
        label, name, cn_name = line.split(",")
        train_labels.append(label.strip())
        name2label[name.strip()] = label.strip()

zsl_train_path = os.path.join(dataset_dir, superclass, "train")
if not os.path.exists(zsl_train_path):
    os.makedirs(zsl_train_path)

for ci in sorted(os.listdir(train_img_path)):
    src_dir = os.path.join(train_img_path, ci)
    shutil.copytree(src_dir, os.path.join(zsl_train_path, ci[2:]))

for name in os.listdir(zsl_train_path):
    label = name2label[name]
    os.rename(os.path.join(zsl_train_path, name), os.path.join(zsl_train_path, label))

labels = []
attrs = []
# attr_w_path = os.path.join(
#     attr_path, "zsl_a_%s_train_annotations_attributes_per_class_20180321.txt" % superclass)
attr_w_path = os.path.join(
    attr_path, "zsl_b_%s_train_annotations_attributes_per_class_20180321.txt" % superclass)

with open(attr_w_path, "r") as fr:
    for line in fr.readlines():
        label, attr = line.split(",")
        labels.append(label.strip())
        attr_v = attr.strip().split(" ")[1:-1]
        attr_v = [np.float64(ai) for ai in attr_v]
        attrs.append(attr_v)

test_labels = list(set(labels).difference(set(train_labels)))
train_labels.sort()
test_labels.sort()
train_ids = [int(label[-2:]) - 1 for label in train_labels]
test_ids = [int(label[-2:]) - 1 for label in test_labels]
train_ids = np.array(train_ids)
test_ids = np.array(test_ids)
np.save("data/%s_train_ids.npy" % superclass, train_ids)
np.save("data/%s_test_ids.npy" % superclass, test_ids)

attr_w = np.array(attrs)
attr_w.shape
np.save("data/%s_attr.npy" % superclass, attr_w)

for label in labels:
    cpath = os.path.join(zsl_train_path, label)
    if not os.path.exists(cpath):
        os.makedirs(cpath)

from sklearn.model_selection import train_test_split

trainval_path = os.path.join(dataset_dir, superclass, "trainval")
tv_train_path = os.path.join(trainval_path, "train")
tv_val_path = os.path.join(trainval_path, "val")

for pdir in [tv_train_path, tv_val_path]:
    if not os.path.exists(pdir):
        os.makedirs(pdir)

for cls_name in labels:
    train_cls_dir = os.path.join(tv_train_path, cls_name)
    val_cls_dir = os.path.join(tv_val_path, cls_name)
    for pdir in [train_cls_dir, val_cls_dir]:
        if not os.path.exists(pdir):
            os.makedirs(pdir)
    class_dir = os.path.join(zsl_train_path, cls_name)
    imgs = os.listdir(class_dir)
    if len(imgs) == 0:
        continue
    X_train, X_val = train_test_split(imgs, test_size=0.2, random_state=42)
    for img in X_train:
        shutil.copy2(os.path.join(class_dir, img), train_cls_dir)
    for img in X_val:
        shutil.copy2(os.path.join(class_dir, img), val_cls_dir)

