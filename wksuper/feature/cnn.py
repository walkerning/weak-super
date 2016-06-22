# -*- coding: utf-8 -*-
"""
CNN特征提取器
"""

import sys
import os
import numpy as np

from skimage.feature import hog
from skimage import color
from skimage.transform import resize

from . import FeatureExtractor

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CNNFeatureExtractor(FeatureExtractor):
    TYPE = "cnn"

    def __init__(self, cfg, **kwargs):
        super(CNNFeatureExtractor, self).__init__(cfg, **kwargs)
        self.proto = self.cfg["proto"]
        self.caffemodel = self.cfg["caffemodel"]

        faster_rcnn_path = self.cfg.get("faster_rcnn_path", os.path.join(_root, "cnn", "py-faster-rcnn", "lib"))
        pycaffe_path = self.cfg.get("pycaffe_path", os.path.join(os.path.dirname(faster_rcnn_path), "caffe-fast-rcnn", "python"))
        sys.path.insert(0, faster_rcnn_path)
        sys.path.insert(0, pycaffe_path)
        # will raise if not make!
        import caffe
        print "Caffe path: ", caffe.__file__

        if self.cfg.get("gpu_mode", False):
            caffe.set_mode_gpu()
            caffe.set_device(self.cfg["gpu_id"])

        self.net = caffe.Net(self.proto, self.caffemodel, caffe.TEST)
        
    def extract_from_rois(self, im, rois):
        """
        Parameters
        ------------
        im: np.array 图片数据
        rois: ROI列表, [(x1, y1, x2, y2)]
        """
        # 设置网络输入
        # 默认test scale为600(短边)
        from fast_rcnn.test import _get_blobs
        blobs, unused_im_scale_factors = _get_blobs(im, rois)
    
        # When mapping from image ROIs to feature map ROIs, there's some aliasing
        # (some distinct image ROIs get mapped to the same feature ROI).
        # Here, we identify duplicate feature ROIs, so we only compute features
        # on the unique subset.
        # dedup_boxes默认为1/16, 对于CaffeNet正确
        if self.cfg["dedup_boxes"] > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(blobs['rois'] * self.cfg["dedup_boxes"]).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            blobs['rois'] = blobs['rois'][index, :]
            #rois = rois[index, :] # ? 这里还有什么用...用于debug?
    
        # reshape network inputs
        self.net.blobs['data'].reshape(*(blobs['data'].shape))
        self.net.blobs['rois'].reshape(*(blobs['rois'].shape))
        features = self.net.forward(data=blobs['data'].astype(np.float32, copy=False),
                                    rois=blobs['rois'].astype(np.float32, copy=False))["pool5"]

        # import ipdb
        # ipdb.set_trace()
        if self.cfg["dedup_boxes"] > 0:
            # Map scores and predictions back to the original set of boxes
            features = features[inv_index, :, :, :].reshape(rois.shape[0], -1)
        return features

    def dim(self):
        return reduce(lambda x,y: x*y, self.net.blobs.values()[-1].data.shape[1:], 1)

    @staticmethod
    def calculate_hog_dim(oris, image_size, pixels_per_cell, cells_per_block):
        assert len(cells_per_block) == 2
        assert len(pixels_per_cell) == 2
        stride = 8
        block_size = (cells_per_block[0] * pixels_per_cell[0],
                      cells_per_block[1] * pixels_per_cell[1])
        return oris * cells_per_block[0] * cells_per_block[1] * ((image_size[0] - block_size[0])/stride + 1) * ((image_size[1] - block_size[1])/stride + 1)
