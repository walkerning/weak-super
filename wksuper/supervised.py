# -*- coding: utf-8 -*-

from .trainer import Trainer
from .dataset import Dataset
from .feature import FeatureExtractor as _FE
from .detector import Detector

class SupervisedTrainer(Trainer):
    """
    强监督训练器
    """
    TYPE = "supervised"
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = Dataset.get_registry(cfg["dataset"]["type"])(cfg)
        self.feat_ext = _FE.get_registry(cfg["feature"]["type"])(cfg)
        self.detectors = []

