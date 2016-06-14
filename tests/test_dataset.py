# -*- coding: utf-8 -*-

cfg_str = """
[dataset]
type = "voc"
path = "/dataset/VOC/VOC2007/VOCdevkit"
year = 2007
use_diff = false
debug = true
"""

import pytest

@pytest.fixture(params=range(20))
def cls(request):
    return request.param

# Failed to patch the decorator...
# from mock import patch
# @patch("wksuper.helper.cache_pickler")
# def null_cache_pickler(_):
#     def wrapper(func):
#         return func
#     return wrapper

from wksuper.config import cfg_from_str
from wksuper.dataset.voc import VOCHandler

class TestVOCHandler(object):
    def setup(self):
        self.cfg = cfg_from_str(cfg_str)
        print self.cfg
        self.dataset = VOCHandler(self.cfg)
        print self.dataset

    def test_load_annotations_for_cls_train(self, cls):
        result = self.dataset.load_annotations_for_cls("train", cls)
        assert len(result) == len(self.dataset.positive_train_indexes(cls))

    def test_load_annotations_for_cls_val(self, cls):
        result = self.dataset.load_annotations_for_cls("val", cls)
        assert len(result) == len(self.dataset.positive_val_indexes(cls))

    def test_load_annotations_for_cls_test(self, cls):
        result = self.dataset.load_annotations_for_cls("test", cls)
        assert len(result) == len(self.dataset.positive_test_indexes(cls))

    def test_boxes_num(self):
        assert self.dataset.train_boxes_num == 6301
        assert self.dataset.val_boxes_num == 6307
        assert self.dataset.test_boxes_num == 12032
