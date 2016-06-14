# -*- coding: utf-8 -*-
"""
数据库处理
"""

from ..exceptions import NotImplementedError
from ..meta import meta

class Dataset(object):
    _ROLE = "dataset handler"
    TYPE = None
    __metaclass__ = meta

    def __init__(self, cfg, **kwargs):
        # should we move all the same initialization code of 
        # base classes into meta.__call__
        self.cfg = cfg["dataset"]
        assert self.cfg["type"] == self.TYPE
        # override configurations using keyword arguments
        for name, value in kwargs.iteritems():
            self.cfg[name] = value

    @property
    def class_names(self):
        """
        返回列表, `K` 类别的名字, 对于一些数据集, 不一定实现此方法
        """
        raise NotImplementedError()

    @property
    def class_number(self):
        """
        返回一个整数 `K`, 代表该数据集一共分多少类
        """
        raise NotImplementedError()

    def train_indexes(self, _cls):
        """
        获得某类别_cls所有训练(train)图像信息
        Returns
        ------------
        train_infos: [(index, label)...]
              每个元素为一个元组, 元组的第一个元素index为图片的索引.
              元组的第二个元素label为-1或者1, 代表此图片对于类别_cls
              为反例图像或者正例图像
        """
        raise NotImplementedError()

    def val_indexes(self, _cls):
        """
        获得某类别_cls所有验证(val)图像信息
        Returns
        ------------
        val_infos: [(index, label)...]
              每个元素为一个元组, 元组的第一个元素index为图片的索引.
              元组的第二个元素label为-1或者1, 代表此图片对于类别_cls
              为反例图像或者正例图像
        """
        raise NotImplementedError()

    def test_indexes(self, _cls):
        """
        获得某类别_cls所有测试(test)图像信息,
        注意有些数据集的test图像的label其实是未知的, 对于这些数据集,
        此接口几乎无用
        Returns
        ------------
        test_infos: [(index, label)...]
              每个元素为一个元组, 元组的第一个元素index为图片的索引.
              元组的第二个元素label为-1, 1或者0, 代表此图片对于类别_cls
              为反例图像或者正例图像, 或者未知
        """
        raise NotImplementedError()

    def positive_train_indexes(self, _cls):
        """
        获得某种类型`_cls`的所有正例图像的索引列表

        Parameters
        ------------
        _type: 类型编号

        Returns
        ------------
        正例图像索引列表
        """
        return [index for (index, l) in self.train_indexes(_cls) if l == 1]

    def negative_train_indexes(self, _cls):
        """
        获得某种类型`_cls`的所有负例图像的索引列表

        Parameters
        ------------
        _type: 类型编号

        Returns
        ------------
        正例图像索引列表
        """
        return [index for (index, l) in self.train_indexes(_cls) if l == -1]

    def positive_val_indexes(self, _cls):
        return [index for (index, l) in self.val_indexes(_cls) if l == 1]

    def positive_test_indexes(self, _cls):
        return [index for (index, l) in self.test_indexes(_cls) if l == 1]

    def get_image_at_index(self, index):
        """
        获得某个索引对应的图像数据

        Returns
        ------------
        im: numpy.array
            图像数据
        """
        raise NotImplementedError()

# import and register all classes
from . import voc
