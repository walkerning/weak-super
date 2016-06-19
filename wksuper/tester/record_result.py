# -*- coding: utf-8 -*-
"""
记录结果的钩子
"""
import time

from .hook import Hook

class RecordResultHook(Hook):
    TYPE = "RecordResult"

    def __init__(self, tester):
        super(RecordResultHook, self).__init__(tester)
        self.record_file_name = "record_{}_{}.txt".format(self.tester.TYPE,
                                                          time.strftime("%F-%H-%M-%S"))
        print "Hook RecordResult: Using record file: {}".format(self.record_file_name)
        self.record_file = open(self.record_file_name, "w")

    def post_test(self, *args):
        im_ind, rois_score = args
        if not rois_score:
            print u"检测结果: 图片 {} 没有任何物体".format(im_ind)
        else:
            print u"检测结果: 图片 {} 有 {} 个检测结果".format(im_ind, len(rois_score))
            self.record(im_ind, rois_score)

    def record(self, im_ind, rois_score):
        for bbox, (det_name, det_score) in rois_score.iteritems():
            self.record_file.write(("{im_ind} {det_name} {det_score}" +
                                    "{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n").format(im_ind=im_ind,
                                                                                        det_name=det_name,
                                                                                        det_score=det_score,
                                                                                        bbox=bbox))
            self.record_file.flush()
        print "写入 {}: 图片 {}".format(self.record_file_name, im_ind)
