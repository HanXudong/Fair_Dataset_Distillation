#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 01:41:02 2019

@author: ilia10000
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

from sklearn.metrics.pairwise import euclidean_distances
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

mode = "label" #selection, label, location, label_and_location
num_points = 2 if mode =="2points" else 3
if mode == "2points":
    granularity=80
else: 
    granularity=50
n_neighbors = num_points*granularity
ims=[]
fig=plt.figure()
ax1 = fig.add_subplot(1,2,1)
axs = [fig.add_subplot(3,2,2), fig.add_subplot(3,2,4)]
if num_points > 2: 
    axs.append(fig.add_subplot(3,2,6))
def animate(j):
    
    
    dd=np.zeros((num_points,2))
    dy=[0,0,0]
    
    if mode == "selection":
        dd[0]=X[j]
        dd[1]=X[50+j]
        dd[2]=X[100+j]
        dy[0]=[1,0,0]
        dy[1]=[0,1,0]
        dy[2]=[0,0,1]
    elif mode == "label":
        dd[0]=[4.1,2.8]
        dd[1]=[5, 2.1]
        dd[2]=[7.6,3.]
        dy[0]=[1,0,0]
        dy[1]=[0+j*0.0025,1-j*0.005,0+j*0.0025]
        dy[2]=[0,0,1]
    elif mode == "location":
        dd[0]=[4.1,2.8]
        dd[1]=[5+0.01*j,2.1+0.01*j]
        dd[2]=[7.6,3.]
        dy[0]=[1,0,0]
        dy[1]=[0,1,0]
        dy[2]=[0,0,1]
    elif mode == "location_and_label":
        dd[0]=[4.1,2.8]
        dd[1]=[5+0.01*j,2.1+0.01*j]
        dd[2]=[7.6,3.]
        dy[0]=[1,0,0]
        dy[1]=[0+j*0.0025,1-j*0.005,0+j*0.0025]
        dy[2]=[0,0,1]
    elif mode == "2points":
        dy=[0,0]
        dd[0]=[4.1+0.005*j,2.8+0.005*j]
        dd[1]=[7.6-0.005*j,3]
        dy[0]=[0.75-j*0.001,0.25+j*0.001,0]
        dy[1]=[0,0.25+j*0.001,0.75-j*0.001]
    #dd[0]=[4.1+0.02*j,2.8+0.02*j]
    #dd[0]=[4.1,2.8]
    #dd[1]=[5.7,2.8]
    #dd[2]=[7.6,3.]
    
    
    if not mode == "true":
        distX=[]
        distY=[]
        for i in range(num_points):
            class0 = int(dy[i][0]*granularity)
            class1 = int(dy[i][1]*granularity)
            class2 = int(granularity-class0-class1)
            distX.append(np.repeat([dd[i]], granularity,axis=0))
            #tempy=
            distY.append(np.repeat(0, class0))
            distY.append(np.repeat(1, class1))
            distY.append(np.repeat(2, class2))
        distX=np.concatenate(distX)
        distY=np.concatenate(distY)
    else:
        distX = X
        distY = y
        dy = [[1,0,0]*50, [0,1,0]*50, [0,0,1]*50]
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF8888', '#88FF88', '#8888FF'])
    cmap_bolder = ListedColormap(['#000000', '#000000', '#000000'])
    colors=['#FFAAAA', '#AAFFAA', '#AAAAFF']
    for weights in ['distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        if mode == "true":
            clf = neighbors.KNeighborsClassifier(len(X), weights=weights)
            clf.fit(X,y)
        else:
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(distX, distY)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax1.clear()
        ax1.pcolormesh(xx, yy, Z, cmap=cmap_light)
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        if not mode =="true": 
            ax1.scatter(distX[:, 0], distX[:, 1], c=distY, cmap=cmap_bolder)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        #plt.title("3-Class classification (k = %i)")
        axs[0].pie(dy[0], colors=colors)
        axs[1].pie(dy[1], colors=colors)
        if num_points > 2 :
            axs[2].pie(dy[2], colors=colors)
import matplotlib.animation as animation
if mode == "selection":
    frames = 4
    interval = 500
elif mode=="2points":
    frames = 190
    interval = 20
elif mode == "true":
    frames = 1
    interval = 1000
else:
    frames = 80
    interval = 20
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False) 
#plt.show()
anim.save('{0}_1.gif'.format(mode),writer='imagemagick') 
