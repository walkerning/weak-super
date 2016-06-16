# -*- coding: utf-8 -*-

from ..exceptions import NotImplementedError
from ..meta import meta

class Proposaler(object):
    _ROLE = "proposal maker"
    TYPE = None
    __metaclass__ = meta

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg["proposal"]
        assert self.cfg["type"] == self.TYPE
        # override configurations using keyword arguments
        for name, value in kwargs.iteritems():
            self.cfg[name] = value

    def make_proposal(self, im):
        """
        对于一张图片提出区域提议

        Parameters
        ------------
        im: 图片数据
        
        Returns
        ------------
        rois: numpy.array
            Rx5矩阵: ROI列表 + confidence

        """
        raise NotImplementedError()