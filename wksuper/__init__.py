# -*- coding: utf-8 -*-

import argparse

def main():
    from .trainer import Trainer
    from . import iterative
    from .config import cfg_from_file
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", help="The path of the config file")
    args = parser.parse_args()
    cfg = cfg_from_file(args.cfg_file)
    trainer = Trainer.get_registry(cfg["trainer"]["type"])(cfg)
    trainer.train()
