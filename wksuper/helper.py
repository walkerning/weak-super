# -*- coding: utf-8 -*-
"""
Helpers for wksuper
"""

import os
import sys
import cPickle
import glob
from functools import wraps
import matplotlib.pyplot as plt

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

def visualize(im, rois_score, im_ind=""):
    fig, ax = plt.subplots(figsize=(12, 12))
    # draw the original image
    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect="equal")

    for bbox, (det_name, det_score) in rois_score.iteritems():
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{} {}'.format(det_name, det_score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections'.format(im_ind)),
                 fontsize=14)
    print "image %s handled, %d boxes ploted" % (im_ind, len(rois_score))
    plt.axis("off")
    plt.tight_layout()
    plt.draw()
    plt.show() # will block by default
