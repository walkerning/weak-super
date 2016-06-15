# -*- coding: utf-8 -*-

import traceback

from ..exceptions import NotImplementedError
from ..meta import meta
from .hook import Hook

class Tester(object):
    _ROLE = "tester"
    TYPE = None
    __metaclass__ = meta

    AVAILABLE_HOOK_POINTS = {"post_test"}

    def __init__(self, cfg):
        self._hooks = {}

    def add_hook(self, hk):
        for hk_point in self.AVAILABLE_HOOK_POINTS:
            self.add_hook_to_point(hk_point, hk)

    def add_hook_to_point(self, hk_point, hk):
        assert hk_point in self.AVAILABLE_HOOK_POINTS
        assert isinstance(hk, Hook)
        assert hasattr(hk, hk_point)
        assert hk.TYPE is not None
        print u"加入 {} Hook: {}".format(hk_point, hk.TYPE)
        self._hooks.setdefault(hk_point, []).append(hk)

    def hook(self, hk_point, *args):
        assert hk_point in self.AVAILABLE_HOOK_POINTS
        execute_num = 0
        for hk in self._hooks.get(hk_point, []):
            try:
                getattr(hk, hk_point)(*args)
            except Exception as e:
                print u"执行 {} Hook {} 出错".format(hk_point, hk.TYPE)
                traceback.print_exc()
            else:
                execute_num += 1
        print u"共执行 {} 个 {} Hook".format(execute_num, hk_point)
        
    def test_all(self):
        for im_ind in self.dataset.all_indexes("test"):
            test_ans = self.test(im_ind)
            hook("post_test", im_ind, test_ans)

    def test(self):
        raise NotImplementedError()
