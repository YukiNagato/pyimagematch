from pyimagematch import brox_liu
from pyimagematch.utils.vis_corrs import show_correspondences
from pyimagematch.brox_liu import BroxLiu
import os
import cv2
import numpy as np


data_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

im1_path = os.path.join(data_root, "car1.jpg")
im2_path = os.path.join(data_root, "car2.jpg")

im1 = cv2.imread(im1_path)
im2 = cv2.imread(im2_path)

broxliu = BroxLiu()
broxliu.alpha = 0.012
broxliu.ratio = 0.75
broxliu.minWidth = 20
broxliu.nOuterFPIterations = 7
broxliu.nInnerFPIterations = 1
broxliu.nSORIterations = 30
vx, vy = broxliu.match(im1, im2)

x = np.linspace(0, im1.shape[1]-1, im1.shape[1])
y = np.linspace(0, im1.shape[0]-1, im1.shape[0])
xv, yv = np.meshgrid(x, y)

nx = xv+vx
ny = yv+vy

matches = np.concatenate([
    xv.reshape((1, -1)),
    yv.reshape((1, -1)),
    nx.reshape((1, -1)),
    ny.reshape((1, -1)),
], axis=0)

matches = matches.transpose()
show_correspondences(im1, im2, matches)

