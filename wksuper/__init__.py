# -*- coding: utf-8 -*-

import argparse

def main():
    from .trainer import Trainer
    from . import iterative
    from . import supervised
    from .config import cfg_from_file
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", help="The path of the config file")
    args = parser.parse_args()
    cfg = cfg_from_file(args.cfg_file)
    trainer = Trainer.get_registry(cfg["trainer"]["type"])(cfg)
    print "MAIN: begin train!"
    trainer.train()

def test_main():
    from .tester import Tester
    from .tester import regular
    from .tester import Hook
    from .config import cfg_from_file
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", help="The path of the config file")
    parser.add_argument("--hook", action="append", help="Add hooks, could be ShowImage, Pickle.", default=[])
    args = parser.parse_args()
    cfg = cfg_from_file(args.cfg_file)
    tester = Tester.get_registry(cfg["tester"]["type"])(cfg)
    for hk_name in args.hook:
        tester.add_hook(Hook.get_registry(hk_name)(tester))
    print "MAIN: begin test!"
    tester.test()
