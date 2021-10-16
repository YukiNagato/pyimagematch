import os
import cv2
import time
from pyimagematch.deepmatching import DeepMatching
from pyimagematch.utils.vis_corrs import show_correspondences

data_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

im1_path = os.path.join(data_root, "climb1.png")
im2_path = os.path.join(data_root, "climb2.png")

im1 = cv2.imread(im1_path, 0).astype('float32')
im2 = cv2.imread(im2_path, 0).astype('float32')

dm = DeepMatching()
dm.dm_params.ngh_rad=128
# dm.dm_params.n_thread = 4
results = dm.matching(im1, im2)
show_correspondences(im1, im2, results)