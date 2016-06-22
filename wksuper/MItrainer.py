# -*- coding: utf-8 -*-

from .trainer import Trainer
from .dataset import Dataset
from .proposal import Proposaler
from .feature import FeatureExtractor as _FE
from .detector import Detector

from collections import OrderedDict
import numpy as np

import misvm

class MITrainer(Trainer):
    """
    MI-SVM Trainer
    """
    TYPE = "miltrainer"

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = Dataset.get_registry(cfg["dataset"]["type"])(cfg)
        self.proposaler = Proposaler.get_registry(cfg["proposal"]["type"])(cfg)
        self.feat_ext = _FE.get_registry(cfg["feature"]["type"])(cfg)

        self.pos_feature_dict = OrderedDict() # key: im_ind
        self.neg_feature_dict = OrderedDict()

    def _proposal_and_features(self, im_ind):
        # handle dataset
        im = self.dataset.get_image_at_index(im_ind) # get a image
        rois = self.proposaler.make_proposal(im) # get many proposals of one image
        # extract features
        return self.feat_ext.extract_from_rois(im, rois[:, :4]) # every proposal has a feature

    def train(self):
        print "'{}' trainer start to train!".format(self.TYPE)

        for cls_ind in range(self.dataset.class_number):
            detector = Detector.get_registry(self.cfg["detector"]["type"])(self.cfg)

            print "start to train class ", self.dataset.class_names[cls_ind]
            pos_train_indexes = self.dataset.positive_train_indexes(cls_ind) # all positive image indexes in every class cls_ind
            neg_train_indexes = self.dataset.negative_train_indexes(cls_ind) # all negative image indexes in every class cls_ind

            # get features of all rois of all positive images in every class
            for im_ind in pos_train_indexes:
                if im_ind not in self.pos_feature_dict:
                    self.pos_feature_dict[im_ind] = self._proposal_and_features(im_ind)

            # get features of all rois of all negative images in every class
            for im_ind in neg_train_indexes:
                if im_ind not in self.neg_feature_dict:
                    self.neg_feature_dict[im_ind] = self._proposal_and_features(im_ind)

            # use MIL-SVM trainer
            # get bags
            bags = pos_feature_dict.values()
            bags.extend(neg_feature_dict.values())
            # get labels
            labels = np.append(np.ones((1, len(pos_feature_dict.keys()))), -np.ones((1, len(neg_feature_dict.keys()))))

            detector.train(bags, labels)
            detector.save_param()
