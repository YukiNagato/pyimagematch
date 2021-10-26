from ..clib import py_brox_liu
import os
import numpy as np

__all__ = ['BroxLiu']
BroxLiu = py_brox_liu.BroxLiu

def match(self, im1, im2):
    """
    Match images with type match
    Args:
        im1: h*w*3 rgb image
        im2: h*w*3 rgb image
    """
    if len(im1.shape) == 2:
        im1 = np.tile(im1[...,None], (1, 1, 3))
    if len(im2.shape) == 2:
        im2 = np.tile(im2[...,None], (1, 1, 3))

    im1 = np.ascontiguousarray(im1.astype('float64'))/255
    im2 = np.ascontiguousarray(im2.astype('float64'))/255
    return self.matching_double(im1, im2)


setattr(BroxLiu, 'match', match)
