# -*- coding: utf-8 -*-

import numpy as np

from .trainer import Trainer
from .dataset import Dataset
from .feature import FeatureExtractor as _FE
from .detector import Detector

from .helper import get_param_dir

class SupervisedTrainer(Trainer):
    """
    强监督训练器
    """
    TYPE = "supervised"
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = Dataset.get_registry(cfg["dataset"]["type"])(cfg)
        self.feat_ext = _FE.get_registry(cfg["feature"]["type"])(cfg)
        self.detector_prefix = self.cfg["trainer"].get("detector_prefix", "det_")

    def train(self):
        for cls_ind in range(self.dataset.class_number):
            detector = Detector.get_registry(self.cfg["detector"]["type"])(self.cfg)
            print "start to train class ", self.dataset.class_names[cls_ind]
            indexes_bbox_mapping = self.dataset.load_annotations_for_cls(cls_ind)
            neg_indexes_bboxes_mapping = self._sample_negative_bboxes()

            positive_features = np.zeros((0, self.feat_ext.dim()))
            for ind, bbox in indexes_bbox_mapping.iteritems():
                im = self.dataset.get_image_at_index(ind)
                positive_features = np.vstack((positive_features, 
                                               self.feat_ext.extract_features(im, 
                                                                              bbox)))
            negative_features = np.zeros((0, self.feat_ext.dim()))
            for ind, bbox in neg_indexes_bbox_mapping.iteritems():
                im = self.dataset.get_image_at_index(ind)
                negative_features = np.vstack((negative_features,
                                              self.feat_ext.extract_features(im, 
                                                                             bbox)))
            detector.train(np.vstack(positive_features, negative_features), np.hstack((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0]))))

            detector.save(os.path.join(get_param_dir(self.TYPE), detector_prefix + str(cls_ind)))

    def _sample_negative_bboxes(self):
        pass
            
        
