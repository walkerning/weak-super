# -*- coding: utf-8 -*-
"""
显示图片钩子
"""

from ..helper import visualize

from .hook import Hook

class ShowImageHook(Hook):
    TYPE = "ShowImage"

    def post_test(self, *args):
        im_ind, rois_score = args
        if not rois_score:
            print u"检测结果: 图片 {} 没有任何物体".format(im_ind)
        else:
            print u"检测结果: 图片 {} 有 {} 个检测结果".format(im_ind, len(rois_score))
            self.visualize(im_ind, rois_score)

    def visualize(self, im_ind, rois_score):
        im = self.tester.dataset.get_image_at_index(im_ind)
        visualize(im, rois_score, im_ind=im_ind)
