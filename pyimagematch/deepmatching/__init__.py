from ..clib import py_deepmatching
import cv2

__all__ = ['DeepMatching']
DeepMatching = py_deepmatching.DeepMatching


def match(self, im1, im2):
    """
    Match images with type match
    """
    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1 = im1.astype('float32')
    im2 = im2.astype('float32')
    return self.matching_float(im1, im2)


setattr(DeepMatching, 'match', match)
