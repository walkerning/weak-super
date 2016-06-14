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
    
