# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 23:01:37 2017

@author: sups
"""

import cv2
import os

path_src = '../../race_d/train/'
path_dest = '../../race_d/deep/train/'
cascPath = '../../haarcascade_frontalface_default.xml'
races = ['indian','chinese']

faceCascade = cv2.CascadeClassifier(cascPath)

for race in races:
    imgs = [f for f in os.listdir(path_src+race) if not os.path.isdir(f)]
    for img in imgs:
        print img
        p = os.path.join(path_src+race,img)
        print p
        img_pre = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        faces = faceCascade.detectMultiScale(
            img_pre,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces)==0:
            continue
        for (x,y,w,h) in faces:
            img_post = img_pre[y:y+h, x:x+w]
            break
        img_post = cv2.resize(img_post, (200,200))
        cv2.imwrite(os.path.join(path_dest+race,img), img_post)
        