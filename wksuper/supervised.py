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
        self.neg_pos_ratio = self.cfg["trainer"]["neg_pos_ratio"]

    def train(self):
        for cls_ind in range(self.dataset.class_number):
            detector = Detector.get_registry(self.cfg["detector"]["type"])(self.cfg)
            print "start to train class ", self.dataset.class_names[cls_ind]
            indexes_bbox_mapping = self.dataset.load_annotations_for_cls("train", cls_ind)
            neg_indexes_bbox_mapping = self._sample_negative_bboxes(cls_ind)

            positive_features = np.zeros((0, self.feat_ext.dim()))
            for ind, bbox in indexes_bbox_mapping.iteritems():
                im = self.dataset.get_image_at_index(ind)
                positive_features = np.vstack((positive_features, 
                                               self.feat_ext.extract_from_rois(im, 
                                                                               bbox)))
            negative_features = np.zeros((0, self.feat_ext.dim()))
            for ind, bbox in neg_indexes_bbox_mapping.iteritems():
                im = self.dataset.get_image_at_index(ind)
                negative_features = np.vstack((negative_features,
                                              self.feat_ext.extract_from_rois(im, 
                                                                              bbox)))
            print "Start to train detector {}, postive features {}, negative features {}".format(self.cfg["detector"]["type"],
                                                                                                 positive_features.shape[0],
                                                                                                 negative_features.shape[0])
            detector.train(np.vstack((positive_features, negative_features)),
                           np.hstack((np.ones(positive_features.shape[0]),
                                      np.zeros(negative_features.shape[0]))))
            print "End train detector {}".format(self.cfg["detector"]["type"])
            import pdb
            pdb.set_trace()
            detector.save(os.path.join(get_param_dir(self.TYPE), detector_prefix + str(cls_ind)))

    def _sample_negative_bboxes(self, cls_ind):
        # fixme: 是不是还应该用什么boudingbox都没有的sample, 纯背景!
        mapping = dict(zip(self.dataset.all_indexes("train"), 
                           self.dataset.load_annotations("train")))
        positive_box_num = len(self.dataset.load_annotations_for_cls("train", cls_ind))

        ori_ratio = float(self.dataset.train_boxes_num - positive_box_num) / positive_box_num
        binomial_prob = min(self.neg_pos_ratio / ori_ratio, 1)

        indexes_bbox_mapping = {}
        for im_ind in self.dataset.all_indexes("train"):
            anno = mapping[im_ind]
            pending_neg_bboxes = anno["boxes"][anno["gt_classes"]!=cls_ind, :]
            mask = np.random.binomial(1, 0.6, pending_neg_bboxes.shape[0])
            indexes_bbox_mapping[im_ind] = pending_neg_bboxes[mask.astype(bool), :]
        print "sampled {} negative boxes for class {}".format(len(indexes_bbox_mapping),
                                                              self.dataset.class_names[cls_ind])
        return indexes_bbox_mapping
