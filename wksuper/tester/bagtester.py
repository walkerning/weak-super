# -*- coding: utf-8 -*-
import os
import glab
import numpy as np

from wksuper.tester import Tester
from wksuper.trainer import Trainer
from wksuper.dataset import Dataset
from wksuper.proposal import Proposaler
from wksuper.feature import FeatureExtractor as _FE
from wksuper.detector import Detector
from wksuper.helper import get_param_dir_by_name
from wksuper.nms import nms

class BagTester(Tester):
    TYPE = "bag"

    def __init__(self, cfg):
        super(BagTester, self).__init__(cfg)
        self.cfg = cfg
        self.dataset = Dataset.get_registry(cfg["dataset"]["type"])(cfg)
        
        self.proposaler = Proposaler.get_registry(cfg["proposal"]["type"])(cfg)
        self.proposal_threshold = self.cfg["tester"]["proposal_threshold"]
        self.max_proposal_per_im = self.cfg["tester"]["max_proposal_per_im"]
    
        self.feat_ext = _FE.get_registry(cfg["feature"]["type"])(cfg)
        self.detector_cls = Detector.get_registry(cfg["detector"]["type"])
        self.pre_nms_threshold = self.cfg["tester"]["pre_nms_threshold"]

        # load detector
        detector_prefix = self.cfg["tester"].get("detector_prefix", "det_")
        detector_path = get_param_dir_by_name(self.cfg["tester"]["param_dir"])
        dets = glob.glob(os.path.join(detector_path, "{}*".format(detector_prefix)))
        det_names = []
        for det_path in dets:
            det_names.append(os.path.basename(det_path)[len(detector_prefix):])
        print det_names
        self.detectors = {det_name:self.detector_cls.load(cfg, det_path)
                          for det_name, det_path in zip(det_names, dets)}
            
        self.debug = self.cfg.get("debug", False)

    def _test(self, im_ind):
        im = self.dataset.get_image_at_index(im_ind)
        
        # get proposal
        rois = self.proposaler.make_proposal(im)
        
        # 按照confidence从大到小排序, 取阈值
        index = np.argsort(rois[:, 4])[-1::-1]
        rois = rois[index, :]
        
        _to_index = np.where(rois[:, 4] < self.proposal_threshold)[0]
        if len(_to_index) == 0:
            _to_index = rois.shape[0]
        else:
            _to_index = _to_index[0]
            rois = rois[:min(_to_index, self.max_proposal_per_im), :]
            
        # 提取特征
        features = self.feat_ext.extract_from_rois(im, rois[:, :4])

        # 开始检测
        for det_name, detector in self.detectors.iteritems():
            labels = detector.test([features]) # results only contain 1 or -1
            if labels[0] == 1:
                print "This image has {}.".format(det_name)

        return labels
