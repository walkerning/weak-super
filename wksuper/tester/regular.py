# -*- coding: utf-8 -*-
import os
import glob
import numpy as np

from wksuper.tester import Tester
from wksuper.trainer import Trainer
from wksuper.dataset import Dataset
from wksuper.proposal import Proposaler
from wksuper.feature import FeatureExtractor as _FE
from wksuper.detector import Detector
from wksuper.helper import get_param_dir
from wksuper.nms import nms

class RegularTester(Tester):
    TYPE = "regular"

    def __init__(self, cfg):
        super(RegularTester, self).__init__(cfg)
        self.cfg = cfg
        self.dataset = Dataset.get_registry(cfg["dataset"]["type"])(cfg)

        self.proposaler = Proposaler.get_registry(cfg["proposal"]["type"])(cfg)
        self.proposal_threshold = self.cfg["tester"]["proposal_threshold"]
        self.max_proposal_per_im = self.cfg["tester"]["max_proposal_per_im"]

        self.feat_ext = _FE.get_registry(cfg["feature"]["type"])(cfg)
        self.detector_cls = Detector.get_registry(cfg["detector"]["type"])
        self.pre_nms_threshold = self.cfg["tester"]["pre_nms_threshold"]

        # 加载detector
        detector_prefix = self.cfg["tester"].get("detector_prefix", "det_")
        detector_path = get_param_dir(self.cfg["tester"]["trainer_type"])
        dets = glob.glob(os.path.join(detector_path, "{}*".format(detector_prefix)))
        det_names = []
        for det_path in dets:
            det_names.append(os.path.basename(det_path)[len(detector_prefix):])
        self.detectors = {det_name:self.detector_cls.load(cfg, det_path) 
                          for det_name, det_path in zip(det_names, dets)}

        self.debug = self.cfg.get("debug", False)

    def _test(self, im_ind):
        # 获得图片
        im = self.dataset.get_image_at_index(im_ind)

        # 提框
        rois = self.proposaler.make_proposal(im)
        # 按照confidence从大到小排序, 取阈值
        index = np.argsort(rois[:, 4])[-1::-1]
        rois = rois[index, :]

        _to_index = np.where(rois[:, 4] < self.proposal_threshold)[0]
        if len(_to_index) == 0:
            _to_index = rois.shape[0]
        else:
            _to_index = _to_index[0]
        rois = rois[:min(_to_index,
                         self.max_proposal_per_im), :]

        # 提取特征
        features = self.feat_ext.extract_from_rois(im, rois[:, :4])

        # 开始检测
        rois_scores = {}
        for det_name, detector in self.detectors.iteritems():
            labels, scores = detector.test(features)
            inds = np.where(np.logical_and(labels == 1, 
                                           scores > self.pre_nms_threshold))[0]
            # 非最大值抑制
            pre_nms_rois = np.hstack((rois[inds, :4], scores[inds][:, np.newaxis]))
            keep = nms(pre_nms_rois, self.cfg["tester"]["nms_iou_threshold"])
            post_nms_rois = pre_nms_rois[keep, :]
            for roi in post_nms_rois:
                rois_scores.setdefault(tuple(roi[:4]), []).append((det_name, roi[4]))

        # 选取最可能的检测结果
        final_rois_score = {}
        for roi, dets in rois_scores.iteritems():
            if self.debug:
                print u"DEBUG: {}: {}".format(roi, dets)
            max_det = max(dets, key=lambda x:x[1])
            final_rois_score[roi] = max_det
        return final_rois_score
