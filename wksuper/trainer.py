# -*- coding: utf-8 -*-
"""
Trainer: 每个Trainer实现了一种弱监督训练流程
"""

import argparse

from .config import cfg_from_file
from .meta import meta
from .exceptions import NotImplementedError
from .dataset import Dataset
from .proposal import Proposaler
from .feature import FeatureExtractor as _FE
from .detector import Detector

class Trainer(object):
    _ROLE = "trainer"
    TYPE = None
    __metaclass__ = meta

    def __init__(self, cfg):
        pass

    def train(self):
        raise NotImplementedError()

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
        self.detector = Detector.get_registry(cfg["detector"]["type"])(cfg)

    def train(self):
        print "'{}' trainer start to train!".format(self.TYPE)

        for cls_ind in range(self.dataset.class_number):
            # handle dataset
            print "start to train class ", self.dataset.class_names[cls_ind]
            pos_train_indexes = self.dataset.positive_train_indexes(cls_ind)
            # make proposals
            for im_ind in pos_train_indexes:
                im = self.dataset.get_image_at_index(im_ind)
                rois = self.proposaler.make_proposal(im)
            # extract features

            # initialization

            # alternative deciding latent labels and training detectors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", help="The path of the config file")
    args = parser.parse_args()
    cfg = cfg_from_file(args.cfg_file)
    trainer = Trainer.get_registry(cfg["trainer"]["type"])(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
