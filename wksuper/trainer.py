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
        self.data_handler = Dataset.get_registry(cfg["dataset"]["type"])(cfg)
        self.proposaler = Proposaler.get_registry(cfg["proposal"]["type"])(cfg)
        self.feat_extractor = _FE.get_registry(cfg["feature"]["type"])(cfg)
        self.detector = Detector.get_registry(cfg["detector"]["type"])(cfg)

    def train(self):
        print "'{}' trainer start to train!".format(self.TYPE)
        
        # for cls_ind in range(self.data_handler.class_number):
            # handle dataset
            # print self.data_handler.class_names[cls_ind]
            # print self.data_handler.positive_train_indexes(cls_ind)
            # make proposals
            
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
