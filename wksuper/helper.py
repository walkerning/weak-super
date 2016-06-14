# -*- coding: utf-8 -*-
"""
Helpers for wksuper
"""

import os
import sys
import cPickle
import glob
from functools import wraps

here = os.path.dirname(os.path.abspath(__file__))

PICKLE_CACHE_DIR = os.path.join(here, "cache")
PARAM_DIR = os.path.join(here, "param")

def cache_pickler(key):
    def wrapper(func):
        func_argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
        @wraps(func)
        def _f(*args, **kwargs):
            formatted_key = key.format(**dict(zip(func_argnames, args)))
            formatted_key = formatted_key.format(**kwargs)
            CACHE_PATH = os.path.join(PICKLE_CACHE_DIR, formatted_key + ".pkl")
            if os.path.exists(CACHE_PATH):
                print "Using cache as result of {}: {}".format(func.func_name,
                                                               CACHE_PATH)
                with open(CACHE_PATH, "r") as f:
                    return cPickle.load(f)
            else:
                ans = func(*args, **kwargs)
                print "Dumping cache as result of {}: {}".format(func.func_name,
                                                                 CACHE_PATH)
                with open(CACHE_PATH, "w") as f:
                    cPickle.dump(ans, f)
                return ans
        return _f
    return wrapper

def clean_cache(key_pattern):
    fnames = glob.glob(os.path.join(PICKLE_CACHE_DIR, key_pattern))
    print "Removing ", fnames
    [os.remove(fname) for fname in fnames]

def clean_cache_main():
    assert len(sys.argv) == 2
    clean_cache(sys.argv[1])

def get_param_dir(_type):
    dir_path = os.path.join(PARAM_DIR, _type)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path) # not handle error here!
    return dir_path

