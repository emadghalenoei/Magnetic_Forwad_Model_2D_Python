# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:14:42 2020

@author: emad ghalenoei
"""

import numpy as np
import math
import os
import shutil
from ModelSpace import ModelSpace
from Mag_Kernel_Expanded import Mag_Kernel_Expanded
import matplotlib.pyplot as plt


plt.close('all')

fpath = os.getcwd()+'\Image'
if os.path.exists(fpath) and os.path.isdir(fpath):
    shutil.rmtree(fpath)
os.mkdir(fpath) 

fpath_loaddesk = os.getcwd()

Ndatapoints = 30;  # number of data points
xs = np.linspace(0,60000, Ndatapoints)  # define x of data points
ys = np.linspace(0,50000, Ndatapoints)  # define y of data points
dis_s = np.sqrt((xs-xs[0])**2 + (ys-ys[0])**2);  # compute distance from the first data point

# model space
Z0 = 0              # the shallower depth in model space  
ZEND = 10000        # the deepest depth in model space 
dZ = 100            # the width of prisms in z direction
Pad_Length = 6000   # padding length to minimize the edge effect

CX = 100            # number of prisms in x axis
CZ = 100            # number of prisms in z axis

Azimuth = math.atan2(xs[-1]-xs[0],ys[-1]-ys[0])
xmodel = np.linspace(xs[0]-Pad_Length*math.sin(Azimuth),xs[-1]+Pad_Length*math.sin(Azimuth),CX)
ymodel = np.linspace(ys[0]-Pad_Length*math.cos(Azimuth) ,ys[-1]+Pad_Length*math.cos(Azimuth) ,CX)
dismodel = np.linspace(dis_s[0]-Pad_Length,dis_s[-1]+Pad_Length,CX)
zmodel = np.linspace(Z0,ZEND,CZ)

X, Z = np.meshgrid(xmodel, zmodel)
Y, Z = np.meshgrid(ymodel, zmodel)
DISMODEL, Z = np.meshgrid(dismodel, zmodel)
    
    
dx=abs(X[0,1]-X[0,0]) 
dy=abs(Y[0,1]-Y[0,0])
dz = abs(Z[1,0]-Z[0,0])
dDis = abs(DISMODEL[0,1]-DISMODEL[0,0])

TrueSUSModel = np.load(fpath_loaddesk+'//'+'TrueSUSModel.npy')  #load or define your true model

I = 90 # inclination
Fe = 43314 #(nT)
Azimuth = math.atan2(xs[-1]-xs[0],ys[-1]-ys[0])
Azimuth = Azimuth *180/math.pi
Kernel_Mag = Mag_Kernel_Expanded(DISMODEL,Z,dis_s,I,Azimuth)
Kernel_Mag = 2*Fe* Kernel_Mag

dT_true = Kernel_Mag @ TrueSUSModel.flatten('F') 

   
# Adding noise
noise_T_level = 0.05
sigma_T_original=noise_T_level*max(abs(dT_true))
noise_T_original = sigma_T_original*np.random.randn(Ndatapoints)
dT_simulated = dT_true + noise_T_original

fig, axe = plt.subplots()

axe.plot(dis_s,dT_simulated, 'k-.',linewidth=2) #row=0, col=0
axe.plot(dis_s,dT_true, 'r--',linewidth=2) #row=0, col=0
axe.set(xlabel='X Profile (km)', ylabel='Magnetic (nT)')
plt.show()
figname = 'Magnetic'
fignum = ''
fig.savefig(fpath+'/'+figname+str(fignum)+'.pdf')
plt.close(fig)    # close the figure window
    