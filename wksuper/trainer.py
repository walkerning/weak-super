# -*- coding: utf-8 -*-
"""
Trainer: 每个Trainer实现了一种弱监督训练流程
"""

from .meta import meta
from .exceptions import NotImplementedError

class Trainer(object):
    _ROLE = "trainer"
    TYPE = None
    __metaclass__ = meta

    def __init__(self, cfg):
        pass

    def train(self):
        raise NotImplementedError()
