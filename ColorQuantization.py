# -*- coding: utf-8 -*-
"""
@author: Himanshu Garg
UBID: 5292195
"""

"""
reference taken from https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
for drawing of ellipses on the scatter plot
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import time
import csv
from matplotlib.patches import Ellipse

s = time.time()

UBID = '50292195'; 
np.random.seed(sum([ord(c) for c in UBID]))

def writeImage(name, img):
    cv2.imwrite(name,img)
    print("\n****" + name + " saved****")
    #print("height:",len(img))
    #print("width:",len(img[0]))

def plotText(pts):
    for pt in pts:
        txt = "(" + str(pt[0]) + "," + str(pt[1]) + ")"
        plt.text(pt[0]+0.035,pt[1],txt,fontsize=8)

def plotMus(muMat,colors):
    for m,c in zip(muMat,colors):
        plt.scatter(m[0],m[1],s=50,c=c,marker='o',edgecolors=c)
    plotText(muMat)

def plotPoints(clusterDict,p,colors):
    for key,d in clusterDict.items():
        d = np.array(d)
        if len(d)>0:
            c = colors[key-1]
            plt.scatter(d[:,0],d[:,1],s=100,facecolor='None',marker='^',edgecolors=c)
    plotText(p)
    
def savePlot(title,filename):
    plt.title(title)
    plt.savefig(filename)
    print("\n****" + filename + " saved****")
    plt.clf()

def computeNewMu(clusterDict):
    mus = []
    for key,value in clusterDict.items():
        N = len(value)
        sm = np.round(np.sum(value,axis = 0)/N,2)
        mus.append(sm)
    
    return mus

def computeNewCovariance(clusterDict,muMat):
    cov = []
    for i in range(len(muMat)):
        cov.append(np.cov(np.transpose(clusterDict[i+1])))
    
    return np.array(cov)

def computeClassification(p,muMat):
    flag = 1
    clusterDict = {}
    clsfVector = []
    for i in p:
        d = []
        for ind,m in enumerate(muMat):
            d.append(np.sqrt(np.sum(np.square(np.subtract(m,i)))))
            if flag == 1: 
                clusterDict[ind+1] = []
        flag = 0
        #minindx = np.argmin(d)
        minindx = d.index(min(d))
        clsfVector.append(muMat[minindx])
        clusterDict[minindx+1].append(i)
    
    return clusterDict,np.array(clsfVector)


def computeGMM(p,muMat,covMat):
    flag = 1
    clusterDict = {}
    pdfds = []
    clsfVector = []
    for ind,m in enumerate(muMat):
        gmmVar = multivariate_normal(mean=m, cov=covMat[ind], allow_singular=False)
        pdfds.append(gmmVar.pdf(p))
        if flag == 1: 
            clusterDict[ind+1] = []
    
    #indices = np.argmax(pdfds, axis=0)
    pdfds = np.array(pdfds)
    maxs = np.max(pdfds,axis=0)
    indices = [pdfds[:,i].tolist().index(maxs[i]) for i in range(pdfds.shape[1])]
    for ind,pt in enumerate(p):
        cs = indices[ind]
        clusterDict[cs+1].append(pt)
        clsfVector.append(muMat[cs])
        
    return clusterDict,clsfVector


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def plotPointsAndEllipse(clusterDict,p,cov,muMat,colors):
    for key,d in clusterDict.items():
        d = np.array(d)
        if len(d)>0:
            c = colors[key-1]
            plt.scatter(d[:,0],d[:,1],s=20,c=c,marker='o',edgecolors=c)
            plot_cov_ellipse(cov[key-1],muMat[key-1], nstd=3, alpha=0.5, color=c)

p = np.array([[5.9,3.2],
             [4.6,2.9],
             [6.2,2.8],
             [4.7,3.2],
             [5.5,4.2],
             [5.0,3.0],
             [4.9,3.1],
             [6.7,3.1],
             [5.1,3.8],
             [6.0,3.0]])

muMat = np.array([[6.2, 3.2],   #Cluster 1
         [6.6, 3.7],            #Cluster 1
         [6.5, 3.0]])           #Cluster 3


print("\n\n#######################TASK 3.1 - 3.3 ##############################")
for i in range(2):
    print("\n--------------------- iteration "+str(i+1)+ " ------------------")
    colors=['r','g','b']
    plotMus(muMat,colors) 
    [clusterDict,clsfVector] = computeClassification(p,muMat)
    plotPoints(clusterDict,p,colors)
    print("\nClassification Vector :\n",clsfVector)
    savePlot("","task3_iter"+str(i+1)+"_a.jpg")
    muMat = computeNewMu(clusterDict)
    print("\nNew Mu Matrix :\n",np.array(muMat))
    plotMus(muMat,colors)
    savePlot("","task3_iter"+str(i+1)+"_b.jpg")
    
print("\n----------------------------------------------------------------------")

print("\n\n########################## TASK 3.4 ################################")
baboon = cv2.imread("baboon.jpg",1)
h,w,d = baboon.shape
babooncpy = baboon.copy()
orgbaboonpts = baboon.reshape(-1,3)
baboonpts = babooncpy.reshape(-1,3)
np.random.shuffle(baboonpts)
Ks = [3,5,10,20]
#Ks = [3]
for k in Ks:
    clsfVector = []
    bmuMat = baboonpts[:k]
    for i in range(20):
        [clusterDict,clsfVector] = computeClassification(orgbaboonpts,bmuMat)
        bmuMat = computeNewMu(clusterDict)
    finalBaboon = clsfVector.reshape(h,w,3)
    writeImage("task3_baboon_"+str(k)+".jpg",finalBaboon)
    print("\n ")

print("\n----------------------------------------------------------------------")

print("\n\n########################## TASK 3.5_a ##############################")
muMat = np.array([[6.2, 3.2],           #Cluster 1
                 [6.6, 3.7],            #Cluster 1
                 [6.5, 3.0]])           #Cluster 3
cov = np.array([[[0.5,0],[0,0.5]],
               [[0.5,0],[0,0.5]],
               [[0.5,0],[0,0.5]]])
[clusterDict,clsfVector] = computeGMM(p,muMat,cov)
muMat = computeNewMu(clusterDict)
print("\nNew Mu Matrix :\n",np.array(muMat))
print("\n----------------------------------------------------------------------")

print("\n\n########################## TASK 3.5_b ##############################")
data = []
muMat = np.array([[4.0, 81],           #Cluster 1
                 [2.0, 57],            #Cluster 1
                 [4.0, 71]])           #Cluster 3
    
cov = np.array([[[1.30,13.98],[13.98,184.82]],
               [[1.30,13.98],[13.98,184.82]],
               [[1.30,13.98],[13.98,184.82]]])

with open('faithful.csv', 'r') as fi:
     reader = csv.reader(fi)
     for indrow,row in enumerate(reader):
         if indrow == 0:
             continue
         data.append(row[1:])

data = np.array(np.float32(data))
colors=['r','g','b']
for i in range(5):
    [clusterDict,clsfVector] = computeGMM(data,muMat,cov)
    plotPointsAndEllipse(clusterDict,data,cov,muMat,colors)
    savePlot("","task3_gmm_iter"+str(i+1)+".jpg")
    muMat = computeNewMu(clusterDict)
    print("\nNew Mu Matrix :\n",np.array(muMat))
    cov = computeNewCovariance(clusterDict,muMat)
print("\n----------------------------------------------------------------------")

end = time.time()
print("Total Elapsed time: ",end-s)