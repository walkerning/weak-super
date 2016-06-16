import numpy as np
from .cpu_nms import cpu_nms

def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    return cpu_nms(dets.astype(np.float32), thresh)
