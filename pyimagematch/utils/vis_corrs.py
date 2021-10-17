import sys
import matplotlib
matplotlib.use('TKAgg')
from PIL import Image
import numpy as np
from matplotlib.pyplot import *


def show_correspondences(img0, img1, corr):
    height, width = img0.shape[:2]
    center = [height/2, width/2]
    color_idxs = np.arctan2(*(corr[:,[1,0]] - center).T)
    color_idxs = np.int32(64*color_idxs/np.pi) % 128
    color_idx_set = set(color_idxs)
    color_idx_dict = {c:i for i, c in enumerate(color_idx_set)}
    color_dict = {m:cm.hsv(i/float(len(color_idx_dict))) for m,i in color_idx_dict.items()}
    
    def motion_notify_callback(event):
      if event.inaxes==None: return
      numaxis = event.inaxes.numaxis
      if numaxis<0: return
      x,y = event.xdata, event.ydata
      ax1.lines = []
      ax2.lines = []
      n = np.sum((corr[:,2*numaxis:2*(numaxis+1)] - [x,y])**2,1).argmin() # find nearest point
      x,y = corr[n,0:2]
      ax1.plot(x,y,'+',ms=10,mew=2,color='blue',scalex=False,scaley=False)
      x,y = corr[n,2:4]
      ax2.plot(x,y,'+',ms=10,mew=2,color='red',scalex=False,scaley=False)
      # we redraw only the concerned axes
      renderer = fig.canvas.get_renderer()
      ax1.draw(renderer)  
      ax2.draw(renderer)
      fig.canvas.blit(ax1.bbox)
      fig.canvas.blit(ax2.bbox)
    
    def noticks():
      xticks([])
      yticks([])
    clf()
    ax1 = subplot(221)
    ax1.numaxis = 0
    imshow(img0,interpolation='nearest')
    noticks()
    ax2 = subplot(222)
    ax2.numaxis = 1
    imshow(img1,interpolation='nearest')
    noticks()
    
    ax = subplot(223)
    ax.numaxis = -1
    imshow(img0/2+64,interpolation='nearest')
    # for idx in range(corr.shape[0]):
    #     plot(corr[idx, 0], corr[idx, 1], '+', ms=10, mew=2, color=color_dict[color_idxs[idx]], scalex=0, scaley=0)
    for m in color_idx_set:
      plot(corr[color_idxs==m,0],corr[color_idxs==m,1],'+',ms=10,mew=2,color=color_dict[m],scalex=0,scaley=0)
    noticks()
    noticks()
    
    ax = subplot(224)
    ax.numaxis = -1
    imshow(img1/2+64,interpolation='nearest')
    # for idx in range(corr.shape[0]):
    #     plot(corr[idx, 2], corr[idx, 3], '+', ms=10, mew=2, color=color_dict[color_idxs[idx]], scalex=0, scaley=0)
    for m in color_idx_set:
      plot(corr[color_idxs==m,2],corr[color_idxs==m,3],'+',ms=10,mew=2,color=color_dict[m],scalex=0,scaley=0)
    noticks()
    
    subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                    wspace=0.02, hspace=0.02)
    
    fig = get_current_fig_manager().canvas.figure
    cid_move = fig.canvas.mpl_connect('motion_notify_event',motion_notify_callback)
    print("Move your mouse over the top images to visualize individual matches")
    show()
    fig.canvas.mpl_disconnect(cid_move)