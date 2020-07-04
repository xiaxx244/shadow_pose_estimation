from __future__ import division
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob
import cv2
import math
from numpy import array
from sklearn.cluster import KMeans
from io import StringIO
from PIL import Image
import math
import face_recognition
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction import image
from sklearn.cluster import DBSCAN
from multiprocessing.dummy import Pool as ThreadPool
from matplotlib import pyplot as plt

def hull(ymin,xmin,ymax,xmax,image):
    img=image[ymin:ymax,xmin:xmax]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blur = cv2.blur(gray, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    temp=[]
    hull=[]
    for i in range(len(contours)):
    # creating convex hull object for each contour
        hull1 = cv2.convexHull(contours[i])
        temp.append(len(cv2.convexHull(contours[i], False).tolist()))
        #cv2.drawContours(img, [hull1], -1, (0, 0, 255), 1)
    for j in temp:
        hull.append(j)
    #print(hull)
    #print(hull)
    #cv2.imwrite("pool_trial4_hull/contours"+str(count)+'-'+str(count1)+'.jpg', img)
    return float(sum(hull)/len(hull))
def get_hist(ymin,xmin,ymax,xmax,image):
    img=image[ymin:ymax,xmin:xmax]
    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    maxl=max(hist[0])
    sum=0
    for j in hist:
        for i in j:
            sum=sum+i
    #print(sum)
    #plt.figure()
    #plt.title("Grayscale Histogram")
    #plt.xlabel("Bins")
    #plt.ylabel("# of Pixels")
    #plt.plot(hist)
    #plt.xlim([0, 256])
    #plt.show()
    return float(sum/((ymax-ymin)*(xmax-xmin)))
def get_average_color(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    #image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    #color=cv2.mean(image)
    image=np.array(image)
    color=[]
    crop=np.array_split(image,4)
    for i in range(4):
        color.append(np.mean(crop[i]))
    #print(color)
    return color
def get_moment(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(image)).flatten()
def remove_inf(ls):
    for x in range(len(ls)):
        for y in range(len(ls[x])):
            for z in range(len(ls[x][y])):
                if ls[x][y][z]==float('-inf'):
                    ls[x][y][z]=0
    return ls
def get_amp(ymin,xmin,ymax,xmax,image):
    image=image[ymin:ymax,xmin:xmax]
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum =remove_inf(20*np.log(np.abs(fshift)))
    #plt.imshow(20*np.log(np.abs(fshift))[1])
    #plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #plt.savefig('pool_trial3_fft.jpg')
    sum=0
    for x in range(len(magnitude_spectrum)):
        for y in range(len(magnitude_spectrum[x])):
            sum=sum+magnitude_spectrum[x][y]
    sum[0]=float(sum[0]/float(((ymax-ymin)*(xmax-xmin))))
    sum[1]=float(sum[1]/float(((ymax-ymin)*(xmax-xmin))))
    sum[2]=float(sum[2]/float(((ymax-ymin)*(xmax-xmin))))
    return sum
def approx_shape(ymin,xmin,ymax,xmax,image):
    pts=[]
    image=image[ymin:ymax,xmin:xmax]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    #cv2.imwrite("pool_trial4_edged/edges-"+str(count)+'-'+str(count1)+'.jpg',edged)
    _,cnts,_=cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        pts.append(len(approx))
        #cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
    #cv2.imwrite('pool_trial4_polygon/result-'+str(count)+'-'+str(count1)+'.jpg',image)
    if len(pts)!=0:
        return float(sum(pts)/(len(pts)))
    else:
        return 0
def face_detector(image,ymin,ymax,xmin,xmax):
    face_locations = []
    face_encodings = []
    face_names = []
    rgb_frame = image[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    if face_locations==[]:
        return (0,0,0,0)
    else:
        return face_locations[0]
def get_five_features(image,ymin,ymax,xmin,xmax):
    feature=[]
    temp=hull(ymin,xmin,ymax,xmax,image)
    feature.append(temp)
    for k in range(len(get_average_color(ymin,xmin,ymax,xmax,image))):
        feature.append(get_average_color(ymin,xmin,ymax,xmax,image)[k])
    for j in get_moment(ymin,xmin,ymax,xmax,image).tolist():
        feature.append(j)
    for m in face_detector(image,ymin,ymax,xmin,xmax):
        #print(type(m))
        feature.append(m)
    return feature
