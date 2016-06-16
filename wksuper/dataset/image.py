# -*- coding: utf-8 -*-
"""
用于测试, 图片作为输入
"""

import os
import cv2

from . import Dataset

class ImageHandler(Dataset):
    """
    文件系统上的目录下的所有文件
    """
    TYPE = "image"
    
    def __init__(self, cfg, **kwargs):
        super(ImageHandler, self).__init__(cfg, **kwargs)
        self.image_path = self.cfg["image_path"]
        self.legal_ext = self.cfg["legal_ext"]
        assert os.path.exists(self.image_path)

    def _file_is_legal(self, fname):
        ext = os.path.splitext(fname)[1]
        return ext and ext[1:].lower() in self.legal_ext

    def all_indexes(self, role):
        assert role == "test"
        indexes = [fname for fname in os.listdir(self.image_path) if self._file_is_legal(fname)]
        return indexes

    def get_image_at_index(self, index):
        _path = os.path.join(self.image_path, index)
        assert os.path.exists(_path)
        return cv2.imread(_path)
        
