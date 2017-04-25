# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:00:46 2017

@author: sups
"""
import os, cv2
import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mp
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum

def write_images_to_lmdb(img_dir, db_name, label):
    for root, dirs, files in os.walk(img_dir, topdown = False):
        if root != img_dir:
            continue
        map_size = 200*200*3*3*len(files)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)
        max_key = env.stat()["entries"]
        for idx, name in enumerate(files):
            X = cv2.imread(os.path.join(root, name))
            X = cv2.resize(X, (100,100))
            #print X
            y = label
            datum = array_to_datum(X,y)
            str_id = '{:08}'.format(max_key+idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())   
    txn.commit()
    env.close()
    print " ".join(["Writing to", db_name, "done!"])

path = '../../race_d/train/'
races = ['indian','chinese']
for i in range(2):
    print i
    write_images_to_lmdb(path+races[i], 'train_cnn', int(i))