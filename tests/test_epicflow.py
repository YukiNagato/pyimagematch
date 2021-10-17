import os
import cv2
import time
import numpy as np
from pyimagematch.deepmatching import DeepMatching
from pyimagematch.epicflow import Epicflow
from pyimagematch.utils.vis_corrs import show_correspondences
from pyimagematch.assets import sed_model_file

data_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

im1_path = os.path.join(data_root, "climb1.png")
im2_path = os.path.join(data_root, "climb2.png")

im1 = cv2.imread(im1_path, 0).astype('float32')
im2 = cv2.imread(im2_path, 0).astype('float32')

dm = DeepMatching()
dm.dm_params.ngh_rad=128
dm.dm_params.n_thread = 4
match_results = dm.match(im1, im2)

edgedetector = cv2.ximgproc.createStructuredEdgeDetection(sed_model_file)
im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
edges = edgedetector.detectEdges(im1_rgb / 255.0)

ef = Epicflow()
vx, vy = ef.match(im1_rgb, im2_rgb, edges, match_results)

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
