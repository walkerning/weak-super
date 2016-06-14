# -*- coding: utf-8 -*-

from .trainer import Trainer
from .dataset import Dataset
from .proposal import Proposaler
from .feature import FeatureExtractor as _FE
from .detector import Detector

class IterativeTrainer(Trainer):
    """
    迭代训练器
    """
    TYPE = "iterative"

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = Dataset.get_registry(cfg["dataset"]["type"])(cfg)
        self.proposaler = Proposaler.get_registry(cfg["proposal"]["type"])(cfg)
        self.feat_ext = _FE.get_registry(cfg["feature"]["type"])(cfg)
        self.detectors = []

        self.feature_dict = {}
        self.rois_dict = {}

    def _proposal_and_features(self, im_ind):
        # handle dataset
        im = self.dataset.get_image_at_index(im_ind) # get a image
        rois = self.proposaler.make_proposal(im) # get many proposals of one image
        # extract features
        return rois, self.feat_ext.extract_from_rois(im, rois) # every proposal has a feature

    def train(self):
        print "'{}' trainer start to train!".format(self.TYPE)

        for cls_ind in range(self.dataset.class_number):
            detector = Detector.get_registry(self.cfg["detector"]["type"])(self.cfg)
            self.detectors.append(detector)

            print "start to train class ", self.dataset.class_names[cls_ind]
            pos_train_indexes = self.dataset.positive_train_indexes(cls_ind)

            # proposal and extract features
            for im_ind in pos_train_indexes:
                if im_ind not in self.feature_dict:
                    self.rois_dict[im_ind], self.feature_dict[im_ind] = self._proposal_and_features(im_ind)

            # initialization(choose one proposal from all proposals of every image)
            feats, labels = self.my_init(self.feature_dict)

            # alternative deciding latent labels and training detectors
            results = None # Detect score列表
            while results is None or not self.judge_converge(results): # FIXME: 传参数
                # input: one proposal with feature every image
                # output: a detector
                detector.train(feats, labels) # 更新params
                results = detector.test(feats) # feats：dict所有的value(可能筛选负例)拼接起来的数组，更新results
            detector.save_param()

    def judge_converge(self):
        # 判断收敛条件是否达到, FIXME: 没有指定参数
        pass

    def my_init(self, feature_dict):
        # initalization, 返回
        # feats: 是一个feature的array
        # labels: 与feats一一对应的label标注: 1/-1
        pass
