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

    def get_one_train_image(self):
        raise NotImplementedError()
        
    def get_one_test_image(self):
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
        
    def get_one_train_image(self):
        pass
        
    def get_one_test_image(self):
        pass
