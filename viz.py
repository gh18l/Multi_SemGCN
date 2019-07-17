
"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

def show3Dpose(channels, ax, add_labels=True): # blue, orange
  """
  Visualize a 3d skeleton

  Args
    channels: 17N * 3 vector.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """
  #mpii dataset
  I   = np.array([0,16,1,1,5,6,2,3,1,15,14,14,11,12,8,9]) # start points
  J   = np.array([16,1,5,2,6,7,3,4,15,14,11,8,12,13,9,10]) # end points
  LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
  color = np.array(['b','g','r','y','k','c','m'])
  # Make connection matrix
  num_person = channels.shape[0]/17
  print("the number of person is %d" % num_person)
  for i in range(num_person): #the number of person
    tmp = channels[i::num_person, :]
    for j in np.arange(len(I)):
      x, y, z = [np.array( [tmp[I[j], k], tmp[J[j], k]] ) for k in range(3)]
      ax.plot(x, y, z, lw=2, c=color[i % len(color)])

  # RADIUS = 750 # space around the subject
  # xroot, yroot, zroot = channels[0,0], channels[0,1], channels[0,2]
  # ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  # ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  # ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

  # Get rid of the ticks and tick labels
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  ax.set_zticklabels([])
  ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)

def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """
  Visualize a 2d skeleton

  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(data_utils.H36M_NAMES)*2, "channels should have 64 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

  fig = plt.figure()



  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  RADIUS = 350 # space around the subject
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")

  ax.set_aspect('equal')
