# -*- coding: utf-8 -*-

import os
import re
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from . import Dataset
from wksuper.helper import cache_pickler

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
        self._class_boxes_dict = {}
        self._class_to_num = dict(zip(self._class_names, xrange(self.class_number)))
        # Dict: {role -> {image_ind -> annotations}}
        self.ind_annotation_mapping = {}

    def name_to_ind(self, cls_name):
        return self._class_to_num[cls_name]

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
        if not "train" in self._class_indexes_dict[_cls]:
            self._class_indexes_dict[_cls]["train"] = self._get_indexes(_cls, "train")
        return self._class_indexes_dict[_cls]["train"]

    def val_indexes(self, _cls):
        if isinstance(_cls, int):
            # 也接受整数下标作为类别, 返回字典序第_cls个类别对应的train indexes
            _cls = self._class_names[_cls]

        if not "val" in self._class_indexes_dict[_cls]:
            self._class_indexes_dict[_cls]["val"] = self._get_indexes(_cls, "val")
        return self._class_indexes_dict[_cls]["val"]

    def test_indexes(self, _cls):
        if isinstance(_cls, int):
            # 也接受整数下标作为类别, 返回字典序第_cls个类别对应的train indexes
            _cls = self._class_names[_cls]

        if not "test" in self._class_indexes_dict[_cls]:
            self._class_indexes_dict[_cls]["test"] = self._get_indexes(_cls, "test")
        return self._class_indexes_dict[_cls]["test"]

    def _get_indexes(self, _cls, role):
        image_set_file = os.path.join(self._data_path, "ImageSets", "Main",
                                      "{}_{}.txt".format(_cls, role))
        assert os.path.exists(image_set_file), \
            "path not exists: {}".format(image_set_file)
        return [self._cvt_record(_r.strip()) for _r in open(image_set_file).read().strip().split("\n")]

    def all_indexes(self, role):
        image_set_file = os.path.join(self._data_path, "ImageSets", "Main",
                                      "{}.txt".format(role))
        assert os.path.exists(image_set_file), \
            "path not exists: {}".format(image_set_file)
        return [r.strip() for r in open(image_set_file).read().strip().split("\n")]        
    def _cvt_record(self, rec):
        _t = re.split("[ ]+", rec, 1)
        assert len(_t) == 2
        _t[1] = int(_t[1])
        return tuple(_t)

    def get_image_at_index(self, index):
        image_path = os.path.join(self._data_path, "JPEGImages",
                                  "{}.jpg".format(index))
        assert os.path.exists(image_path), \
            "path not exists: {}".format(image_path)
        return cv2.imread(image_path)

    @property
    def train_boxes_num(self):
        num = 0
        for anno in self.load_annotations("train"):
            num += len(anno["gt_classes"])
        return num

    @property
    def val_boxes_num(self):
        num = 0
        for anno in self.load_annotations("val"):
            num += len(anno["gt_classes"])
        return num

    @property
    def test_boxes_num(self):
        num = 0
        for anno in self.load_annotations("test"):
            num += len(anno["gt_classes"])
        return num

    def load_annotations_for_cls(self, role, _cls):
        if isinstance(_cls, int):
            # 也接受整数下标作为类别, 返回字典序第_cls个类别对应的train indexes
            _cls = self._class_names[_cls]
        key = _cls + "_" + role
        if not key in self._class_boxes_dict:
            self._class_boxes_dict[key] = self._load_annotations_for_cls(role, _cls)
        return self._class_boxes_dict[key]

    @cache_pickler("voc_annotations_{role}_{_cls}")
    def _load_annotations_for_cls(self, role, _cls):
        """
        加载一个类别的positive bbox标注
        QUESTION: 是不是只要在load_annotations包一层. 负例用什么这里是不是不要管
        NOTE: 为了缓存考虑, 这里的_cls为类型名, 不是index
        """
        mapping = dict(zip(self.all_indexes(role), self.load_annotations(role)))
       
        method = getattr(self, "positive_" + role + "_indexes")
        result_dict = {}
        cls_num = self._class_to_num[_cls]
        for ind in method(_cls):
            result_dict[ind] = mapping[ind]["boxes"][mapping[ind]["gt_classes"] == cls_num, :]
            if self.cfg.get("debug", False):
                print "图片 {} 中有 {} 个类型为 {} 的框".format(ind, len(result_dict[ind]), _cls)
        return result_dict

    def load_annotations(self, role):
        if not role in self.ind_annotation_mapping:
            self.ind_annotation_mapping[role] = self._load_annotations(role)
        return self.ind_annotation_mapping[role]

    @cache_pickler("all_voc_annotations_{role}")
    def _load_annotations(self, role):
        """
        加载所有标注
        role: str
            可以为"train", "val", "test"中的一个
        """
        return [self.load_annotation_at_index(image_ind)
                for image_ind in self.all_indexes(role)]

    # Ref: https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py
    def load_annotation_at_index(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.cfg['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        #overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        #seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1


            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_num[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            #overlaps[ix, cls] = 1.0
            #seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        #overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes}

