# -*- coding: utf-8 -*-

import os
import cv2
from . import Dataset

class VOCHandler(Dataset):
    """
    Pascal VOC数据集处理

    TODO
    ------------
    @walkerning: 实现之
    """

    TYPE = "voc"

    def __init__(self, cfg, **kwargs):
        super(VOCHandler, self).__init__(cfg, **kwargs)
        print "Use VOC devkit path: ", self.cfg["path"]
        self._devkit_path = self.cfg["path"]
        self._data_path = os.path.join(self._devkit_path,
                                       "VOC" + str(self.cfg["year"]))
        self._class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                             "bus", "car", "cat", "chair", "cow",
                             "diningtable", "dog", "horse", "motorbike", "person",
                             "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self._class_indexes_dict = {_cls:{} for _cls in self._class_names}

    @property
    def class_number(self):
        return len(self._class_names)

    @property
    def class_names(self):
        return self._class_names

    def train_indexes(self, _cls):
        if isinstance(_cls, int):
            # 也接受整数下标作为类别, 返回字典序第_cls个类别对应的train indexes
            _cls = self._class_names[_cls]

        return self._class_indexes_dict[_cls].setdefault("train", self._get_indexes(_cls, "train"))

    def val_indexes(self, _cls):
        if isinstance(_cls, int):
            # 也接受整数下标作为类别, 返回字典序第_cls个类别对应的train indexes
            _cls = self._class_names[_cls]

        return self._class_indexes_dict[_cls].setdefault("val", self._get_indexes(_cls, "val"))

    def test_indexes(self, _cls):
        if isinstance(_cls, int):
            # 也接受整数下标作为类别, 返回字典序第_cls个类别对应的train indexes
            _cls = self._class_names[_cls]

        return self._class_indexes_dict[_cls].setdefault("test", self._get_indexes(_cls, "test"))

    def _get_indexes(self, _cls, role):
        image_set_file = os.path.join(self._data_path, "ImageSets", "Main",
                                      "{}_{}.txt".format(_cls, role))
        assert os.path.exists(image_set_file), \
            u"图像索引文件不存在: {}".format(image_set_file)
        return [self._cvt_record(_t) for _t in open(image_set_file).read().strip().split("\n")]

    def _cvt_record(self, _t):
        assert len(_t) == 2
        _t[1] = int(_t[1])
        return tuple(_t)

    def get_image_at_index(self, index):
        image_path = os.path.join(self._data_path, "JPEGImages",
                                  "{}.JPEG".format(index))
        assert os.path.exists(image_path), \
            u"图像文件不存在: {}".format(image_path)
        return cv2.imread(image_path)
