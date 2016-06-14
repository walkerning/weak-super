# -*- coding: utf-8 -*-
"""
Hog特征提取器
"""

import numpy as np
from skimage.feature import hog
from skimage import color
from skimage.transform import resize

from . import FeatureExtractor

class HogFeatureExtractor(FeatureExtractor):
    TYPE = "hog"

    def __init__(self, cfg, **kwargs):
        print self
        super(HogFeatureExtractor, self).__init__(cfg, **kwargs)

    def extract_from_rois(self, im, rois):
        """
        Parameters
        ------------
        im: np.array 图片数据
        rois: ROI列表, [(x1, y1, width, height)]
        """
        return self.extract_from_patches([im[x1:x1+width, y1:y1+height] for (x1, y1, width, height) in rois])

    def extract_from_patches(self, patches):
        """
        TODO
        ------------
        Parameters
        ------------
        patchs: 图像区域列表, R个np.array

        Returns
        ------------
        fds: np.array
            特征描述向量矩阵 RxD
        """
        feature_dim = self._calculate_hog_dim()
        print "Hog feautre will be of dim {}".format(feature_dim)

        image_size = (self.cfg["win_size"], self.cfg["win_size"])
        orientations = self.cfg["orientations"]
        pixels_per_cell = (self.cfg["pixels_per_cell"], self.cfg["pixels_per_cell"])
        cells_per_block = (self.cfg["cells_per_block"], self.cfg["cells_per_block"])

        fds = np.zeros((len(patches), feature_dim))
        for i in range(len(patches)):
            # resize patch
            patch = patches[i]
            resized_patch = color.rgb2gray(resize(patch, image_size))
            fd = hog(resized_patch, orientations=orientations,
                     pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block)
            fds[i, :] = fd
        return fds

    def dim(self):
        image_size = (self.cfg["win_size"], self.cfg["win_size"])
        orientations = self.cfg["orientations"]
        pixels_per_cell = (self.cfg["pixels_per_cell"], self.cfg["pixels_per_cell"])
        cells_per_block = (self.cfg["cells_per_block"], self.cfg["cells_per_block"])

        return self.calculate_hog_dim(orientations,
                                      image_size,
                                      pixels_per_cell,
                                      cells_per_block)

    @staticmethod
    def calculate_hog_dim(oris, image_size, pixels_per_cell, cells_per_block):
        assert len(cells_per_block) == 2
        assert len(pixels_per_cell) == 2
        stride = 8
        block_size = (cells_per_block[0] * pixels_per_cell[0],
                      cells_per_block[1] * pixels_per_cell[1])
        return oris * cells_per_block[0] * cells_per_block[1] * ((image_size[0] - block_size[0])/stride + 1) * ((image_size[1] - block_size[1])/stride + 1)
