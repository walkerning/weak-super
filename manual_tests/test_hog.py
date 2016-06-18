# -*- coding: utf-8 -*-

import cv2

from skimage import color
from skimage.transform import resize
from skimage.feature import hog

from wksuper.config import cfg_from_str
from wksuper.feature import FeatureExtractor

cfg_str = """
[feature]
type = "hog"
win_size = 256
orientations = 9
pixels_per_cell = 8
cells_per_block = 3
"""

cfg = cfg_from_str(cfg_str)
fe = FeatureExtractor.get_registry("hog")(cfg)
im = cv2.imread("/dataset/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/000002.jpg")

cv2.imshow("original image", im)

image_size = (fe.cfg["win_size"], fe.cfg["win_size"])
orientations = fe.cfg["orientations"]
pixels_per_cell = (fe.cfg["pixels_per_cell"], fe.cfg["pixels_per_cell"])
cells_per_block = (fe.cfg["cells_per_block"], fe.cfg["cells_per_block"])
resized_patch = color.rgb2gray(resize(im, image_size))
cv2.imshow("resized image", resized_patch)

fd, f_im = hog(resized_patch, orientations=orientations,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               visualise=True)
# import ipdb
# ipdb.set_trace()
cv2.imshow("feature image", f_im*8)
cv2.waitKey(0)
