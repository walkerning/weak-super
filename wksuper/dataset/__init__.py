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
    def class_number(self):
        """
        返回一个整数 `K`, 代表该数据集一共分多少类
        """
        raise NotImplementedError()

    def all_train_indexes(self):
        """
        返回所有训练(train)图像的索引列表 + 一一对应的标注列表
        Returns
        ------------
        inds: numpy.array
              1 x N. 每个元素为一张图像的索引. 
        array_of_labels: numpy.array
              N x K. 每行为一张图像的K类的0\1标注
        """
        raise NotImplementedError()

    def all_val_indexes(self):
        """
        返回所有验证(val)图像的索引列表 + 一一对应的标注列表
        Returns
        ------------
        inds: numpy.array
              1 x N. 每个元素为一张图像的索引. 
        array_of_labels: numpy.array
              N x K. 每行为一张图像的K类的0\1标注
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
        raise NotImplementedError()

    def get_image_at_index(self, index):
        """
        获得某个索引对应的图像数据和对应的弱标注

        Returns
        ------------
        im: numpy.array 
            图像数据
        labels: numpy.array
            1 x K, 每个元素为1或0, 第k个元素代表此图片中有没有第k类物体
        """
        raise NotImplementedError()


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
        print "Use VOC dataset path: ", self.cfg["path"]
