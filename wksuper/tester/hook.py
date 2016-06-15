# -*- coding: utf-8 -*-

from ..meta import meta

class Hook(object):
    TYPE = None
    _ROLE = "hook"
    __metaclass__ = meta
    def __init__(self, tester):
        self.tester = tester

    def post_test(self, *args):
        pass

    def post_test_all(self, *args):
        pass
