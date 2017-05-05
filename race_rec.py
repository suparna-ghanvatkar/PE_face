from skimage import feature
import numpy as np
import cv2
import os
import copy

path = '../race_d/train/'
cascPath = '../haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascPath)
recognizer = cv2.face.createLBPHFaceRecognizer()

dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

feats = []
labels = []
for d in dirs:
    lab = d
    d = path+d+'/'
    image_paths = [os.path.join(d,f) for f in os.listdir(d)]

    for image_path in image_paths:
        #print image_path
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
            #print img.size
            break
        img = cv2.resize(img, (200,200))
        #z = img.reshape((-1,3))
        #z = np.float32(z)
        #lbp = feature.local_binary_pattern(img, 24, 8, method="uniform")
        #(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0,27), range=(0,26))
        #hist = hist.astype("float")
        #hist /= (hist.sum()+ 1e-7)
        feats.append(img)
        if lab=="indian":
            labels.append(0)
        else:
            labels.append(1)

print labels
recognizer.train(feats,np.array(labels))
'''
path = '../race_d/test/'
targ = '../race_d/recognizer_results/'
imgno = 0

image_paths = [os.path.join(path,f) for f in os.listdir(path)]

for image_path in image_paths:
    print image_path
    img1 = cv2.imread(image_path)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        img = img[y:y+h, x:x+w]
        break
    img = cv2.resize(img, (200,200))
    #z = img.reshape((-1,3))
    #z = np.float32(z)
    #lbp = feature.local_binary_pattern(img, 24, 8, method="uniform")
    #(hist,_) = np.histogram(lbp.ravel(), bins=np.arange(0,27), range=(0,26))
    #hist = hist.astype("float")
    #hist /= (hist.sum()+ 1e-7)
    result = cv2.face.MinDistancePredictCollector()
    recognizer.predict(img,result,0)
    pred = result.getLabel()
    if pred==0:
        pred = "Indian"
    else:
        pred = "Chinese"

    imgno = imgno + 1
    cv2.putText(img1, pred, (10,30), cv2.FONT_ITALIC, 1.0, (0,0,255),3)
    cv2.imwrite(targ+str(imgno)+'.png', img1)
    cv2.waitKey(0)
'''
