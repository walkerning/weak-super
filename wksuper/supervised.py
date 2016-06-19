# -*- coding: utf-8 -*-
import os
import numpy as np

from .trainer import Trainer
from .dataset import Dataset
from .feature import FeatureExtractor as _FE
from .detector import Detector

from .helper import get_param_dir, cache_pickler

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
        self.debug = self.cfg["trainer"].get("debug", False)

    def train(self):
        for cls_ind in range(self.dataset.class_number):
            cls_name = self.dataset.class_names[cls_ind]
            detector = Detector.get_registry(self.cfg["detector"]["type"])(self.cfg)
            print "start to train class ", cls_name
            indexes_bbox_mapping = self.dataset.load_annotations_for_cls("trainval", cls_ind)
            neg_indexes_bbox_mapping = self._sample_negative_bboxes(cls_name,
                                                                    self.neg_pos_ratio)

            positive_features = np.zeros((0, self.feat_ext.dim()))
            for ind, bbox in indexes_bbox_mapping.iteritems():
                im = self.dataset.get_image_at_index(ind)
                positive_features = np.vstack((positive_features, 
                                               self.feat_ext.extract_from_rois(im, 
                                                                               bbox)))
                if self.cfg["trainer"].get("flipped", False):
                    # flipped
                    width = im.shape[0]
                    im = im[:, ::-1, :] # flipped image
                    flipped_bbox = bbox.copy()
                    oldx1 = flipped_bbox[:, 0].copy()
                    oldx2 = flipped_bbox[:, 2].copy()
                    flipped_bbox[:, 0] = width - oldx2 - 1
                    flipped_bbox[:, 2] = width - oldx1 - 1
                    positive_features = np.vstack((positive_features, 
                                                   self.feat_ext.extract_from_rois(im, 
                                                                                   flipped_bbox)))
            print u"提取postive框feature结束"

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
                                      -np.ones(negative_features.shape[0]))))
            print "End train detector {}".format(self.cfg["detector"]["type"])
            detector.save(os.path.join(get_param_dir(self.TYPE), self.detector_prefix + str(cls_name)))

    @cache_pickler("negative_indexes_bbox_mapping_{cls_name}_{neg_pos_ratio}")
    def _sample_negative_bboxes(self, cls_name, neg_pos_ratio):
        # fixme: 是不是还应该用什么boudingbox都没有的sample, 纯背景!
        mapping = dict(zip(self.dataset.all_indexes("trainval"), 
                           self.dataset.load_annotations("trainval")))
        positive_box_num = sum([len(v) for v in self.dataset.load_annotations_for_cls("trainval", cls_name).values()])

        ori_ratio = float(self.dataset.trainval_boxes_num - positive_box_num) / positive_box_num
        if self.debug:
            print u"对于类别 {}, 原始数据集中负框/正框: {}".format(cls_name, ori_ratio)
        binomial_prob = min(neg_pos_ratio / ori_ratio, 1)
        if self.debug:
            print "binomial_prob: {}".format(binomial_prob)

        indexes_bbox_mapping = {}
        boxes_num = 0
        for im_ind in self.dataset.all_indexes("trainval"):
            anno = mapping[im_ind]
            pending_neg_bboxes = anno["boxes"][anno["gt_classes"]!=self.dataset.name_to_ind(cls_name), :]
            mask = np.random.binomial(1, binomial_prob, pending_neg_bboxes.shape[0])
            new_boxes_num = np.sum(mask)
            boxes_num += new_boxes_num
            if new_boxes_num:
                pending_neg_bboxes = pending_neg_bboxes[mask.astype(bool), :]
                indexes_bbox_mapping[im_ind] = pending_neg_bboxes
            
        print "sampled {} negative boxes for class {}".format(boxes_num,
                                                              cls_name)
        return indexes_bbox_mapping
