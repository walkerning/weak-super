# -*- coding: utf-8 -*-
"""
由于很多proposal方法都是Matlab或者C++实现,
对于C++可以用boost库 + cv.Mat <-> 的to/from_python_converter
但是统一起来都可以打印啊!!! hhhh 所以这里实现一个统一的类
"""

import os
import subprocess
import tempfile
import shlex
import numpy as np
import cv2
import psutil

from wksuper.proposal.proposaler import Proposaler

IMAGE_FNAME_PLACEHOLDER = "input_image_fname"

# TODO: 配置如何传递到proposaler里面. 通过统一的配置文件或者环境变量? 后者对于c++程序要简单很多
# 不过还是统一实现一个配置文件感觉比较好诶
class SubprocessProposaler(Proposaler):
    TYPE = "subprocess"
    TMP_EXT = "jpg"

    def __init__(self, cfg, **kwargs):
        super(SubprocessProposaler, self).__init__(cfg, **kwargs)
        self.command = self.cfg["command"]
        if not IMAGE_FNAME_PLACEHOLDER in self.command:
            self.command += " \{{}\}".format(IMAGE_FNAME_PLACEHOLDER)
        print "Subprocess proposaler: using command: ", self.command
        self.tmp_file_suffix = "wksuper_proposal.{}".format(self.TMP_EXT)
        self.proc = psutil.Process()

    def parse_output(self, output):
        output = output.strip().split("\n")
        num = len(output)
        rois = np.zeros(shape=(num, 5))
        for i in range(num):
            line = output[i]
            ans = line.strip().split(" ")
            assert len(ans) == 5
            rois[i, :] = np.array([float(x) for x in ans])
        return rois

    def make_proposal(self, im):
        _, fname = tempfile.mkstemp(suffix=self.tmp_file_suffix)
        cv2.imwrite(fname, im)

        # close temporary files to avoid opening too much files, cv2.imwrite do not close file by default
        for f in self.proc.open_files():
            if f.path == fname:
                os.fdopen(f.fd).close()

        output = subprocess.check_output(shlex.split(self.command.format(**{IMAGE_FNAME_PLACEHOLDER:fname})))
        return self.parse_output(output)
