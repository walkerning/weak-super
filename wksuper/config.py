# -*- coding: utf-8 -*-

import toml

def cfg_from_file(file_name):
    return toml.load(file_name)

def cfg_from_str(cfg_str):
    return toml.loads(cfg_str)

if __name__ == "__main__":
    cfg_str = """
[detector]
type = "svm"
kernel = "gaussian"
"""
    cfg = cfg_from_str(cfg_str)
    assert cfg["detector"]["type"] == "svm"
    assert cfg["detector"]["kernel"] == "gaussian"

    import sys
    if sys.argv >= 2:
        cfg = cfg_from_file(sys.argv[1])
        print cfg
