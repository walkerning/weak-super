# -*- coding: utf-8 -*-
"""
二分类器
"""

import cPickle

from ..exceptions import NotImplementedError
from ..meta import meta

from sklearn import svm
import numpy as np

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
        labels: 与features一一对应的标签, 1或-1或0(正例中的负feature和负例中的feature要区别考虑)
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

    def save(self, file_name):
        """
        将detector参数存入文件
        Parameters
        ------------
        file_name: 保存参数的文件名
        """
        raise NotImplementedError()

    @classmethod
    def load(self, cfg, file_name):
        """
        从文件加载detector参数
        Parameters
        ------------
        file_name: 要加载参数的文件名
        """
        raise NotImplementedError()

class SVMDetector(Detector):
    """
    SVM分类器 Kernel: linear
    """
    TYPE = "svm"

    def __init__(self, cfg, **kwargs):
        super(SVMDetector, self).__init__(cfg, **kwargs)
        self.clf = svm.SVC(kernel='linear', probability=True) # use linear-kernel

    def train(self, features, labels):
        # features is a np.array
        self.clf.fit(features, labels) # get the parameters
        return self.clf

    def test(self, features):
        # features is a np.array
        # Attention: If dataset is very small, the results of predict and predict_proba have big differences. 
        labels = self.clf.predict(features)
        return labels, self.clf.predict_proba(features)[np.hstack((((1-labels)/2)[:, np.newaxis],
                                                                   ((labels+1)/2)[:, np.newaxis])).astype(bool)] # return label, probabilities both np.array

    def save(self, file_name):
        with open(file_name, "w") as f:
            cPickle.dump(self.clf, f)

    @classmethod
    def load(cls, cfg, file_name):
        self = cls(cfg)
        with open(file_name, "r") as f:
            self.clf = cPickle.load(f)
        return self
