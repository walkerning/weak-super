# -*- coding: utf-8 -*-
"""
二分类器
"""

from ..exceptions import NotImplementedError
from ..meta import meta

class Detector(object):
    _ROLE = "detector"
    TYPE = None
    __metaclass__ = meta

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg["detector"]
        assert self.cfg["type"] == self.TYPE
        for name, value in kwargs.iteritems():
            self.cfg[name] = value

    def train(self, features, labels):
        """
        Parameters
        ------------
        features: 用于训练的特征列表
        labels: 与features一一对应的标签
        """
        raise NotImplementedError()

    def test(self, features):
        """
        Parameters
        ------------
        features: 用于test的特征列表

        Returns
        ------------
        results: Detect score列表
        """
        raise NotImplementedError()

class SVMDetector(Detector):
    """
    SVM分类器

    TODO
    ------------
    @criminalking 实现之~
    """
    TYPE = "svm"
    
    def __init__(self, cfg, **kwargs):
        super(SVMDetector, self).__init__(cfg, **kwargs)


    