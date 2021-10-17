from ..clib import py_epicflow
import cv2
import os
import numpy as np

__all__ = ['Epicflow']
Epicflow = py_epicflow.Epicflow

def match(self, im1, im2, edges, matches):
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

    im1 = np.ascontiguousarray(im1.astype('float32'))
    im2 = np.ascontiguousarray(im2.astype('float32'))
    
    edges = np.ascontiguousarray(edges.astype('float32'))
    matches = np.ascontiguousarray(matches.astype('float32'))
    return self.matching_float(im1, im2, edges, matches)


setattr(Epicflow, 'match', match)
