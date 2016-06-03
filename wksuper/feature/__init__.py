# -*- coding: utf-8 -*-
"""
特征提取器
"""

from ..exceptions import NotImplementedError
from ..meta import meta

class FeatureExtractor(object):
    _ROLE = "feature extractor"
    TYPE = None
    __metaclass__ = meta

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg["feature"]
        assert self.cfg["type"] == self.TYPE
        # override configurations using keyword arguments
        for name, value in kwargs.iteritems():
            self.cfg[name] = value
        
    def extract_from_rois(self, im, rois):
        """
        Parameters
        ------------
        im: 图片数据
        rois: ROI列表
        """
        raise NotImplementedError()

    def extract_from_patches(self, patches):
        """
        Parameters
        ------------
        patches: 图片区域列表
        """
        raise NotImplementedError()

class HogFeatureExtractor(FeatureExtractor):
    TYPE = "hog"

    def __init__(self, cfg, **kwargs):
        super(HogFeatureExtractor, self).__init__(cfg, **kwargs)

    def extract_from_rois(self, im, rois):
        """
        Parameters
        ------------
        im: 图片数据
        rois: ROI列表, [(x1, y1, width, height)]
        """
        
        return self.extract_from_patches(im[x1:x1+width, y1:y1+height] for (x1, y1, width, height) in rois)

    def extract_from_patches(self, patches):
        """
        TODO
        ------------
        @criminalking 负责实现~
        """
        raise NotImplementedError()
