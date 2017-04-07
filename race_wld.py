# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 18:42:01 2017

@author: sups
"""

from sklearn.svm import LinearSVR
import numpy as np
import os

label = {'chinese':0,'indian':1}

path = "../race_d/wld_feat/train/"
dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

feats = []
labels = []
for d in dirs:
    lab = d
    d = path+d+'/'
    feat_path = d+'feats'
    feat = open(feat_path,'r')
    for line in feat:
        feats.append(map(float, line.split()))
        labels.append(label[lab])
print labels
wld_feats = np.array(feats)
model = LinearSVR()
model.fit(feats,labels)

print "fitted"
feat.close()
test_path = "../race_d/wld_feat/test/"
dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
count = 0
succ = 0
for d in dirs:
    lab = d
    d = test_path+d+'/'
    feat_path = d+'feats'
    print feat_path
    tfeat = open(feat_path,'r')
    test_feat = []
    for tline in tfeat:
        test_feat = map(float, tline.split())
        count = count+1
        #print feat_path,count
        #print np.array( test_feat)
        print model.predict(np.array(test_feat).reshape(1,-1))
        if model.predict(np.array(test_feat).reshape(1,-1))<=0.3 and lab=='chinese':
            succ = succ+1
        elif model.predict(np.array(test_feat).reshape(1,-1))>0.3 and lab=='indian':
            succ = succ+1
        else:
            print 'incorrect'
    #print model.predict(np.array(test_feat[0]).reshape(1,-1))
print succ,' ',count
'''

    for image_path in image_paths:
        print image_path
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces = faceCascade.detectMultiScale(
            img,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces)==0:
            continue
        for (x,y,w,h) in faces:
            img = img[y:y+h, x:x+w]
            print img.size
            break
        img = cv2.resize(img, (200,200))
        #z = img.reshape((-1,3))
        #z = np.float32(z)
        lbp = feature.local_binary_pattern(img, 24, 8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0,27), range=(0,26))
        hist = hist.astype("float")
        hist /= (hist.sum()+ 1e-7)
        feats.append(hist)
        if lab=="indian":
            labels.append("Indian")
        else:
            labels.append("Chinese")

print labels

'''
