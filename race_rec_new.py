from skimage import feature
import numpy as np
import cv2
import os

path = '../race_d/cropped/train/'

recognizer = cv2.face.createLBPHFaceRecognizer()

dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

feats = []
labels = []
for d in dirs:
    lab = d
    d = path+d+'/'
    image_paths = [os.path.join(d,f) for f in os.listdir(d) if f!='Filenms.txt']

    for image_path in image_paths:
        print image_path
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (180,180))
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

#print labels
recognizer.train(feats,np.array(labels))

path = '../race_d/cropped/test/'
targ = '../race_d/new_results/lbpfacerec/'
imgno = 0
correct = 0

dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

feats = []
labels = []
for d in dirs:
    lab = d
    d = path+d+'/'

    image_paths = [os.path.join(d,f) for f in os.listdir(d) if f!='Filenms.txt']

    for image_path in image_paths:
        print image_path
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (180,180))
        result = cv2.face.MinDistancePredictCollector()
        recognizer.predict(img,result,0)
        pred = result.getLabel()
        if pred==0 and lab=="indian":
            correct += 1
        elif pred==1 and lab=="chinese":
            correct += 1
        else:
            cv2.imwrite(targ+str(imgno)+'.png', img)
        imgno = imgno + 1
        #cv2.putText(img1, pred, (10,30), cv2.FONT_ITALIC, 1.0, (0,0,255),3)
        cv2.waitKey(0)
print correct, ',',imgno
