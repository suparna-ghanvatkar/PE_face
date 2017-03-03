import cv2
import numpy as np
import os

names = {2:'Ritika', 3:'Suparna', 4:'Nisha', 5:'Harsha', 6:'Eshita', 7:'Sharda', 8:'Soumya'}

cascPath = '../haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
path = '../faces/'

recognizer = cv2.face.createLBPHFaceRecognizer()

image_paths = [os.path.join(path, f) for f in os.listdir(path)]

images = []
labels = []

for image_path in image_paths:
    #image_pil = Image.open(image_path).convert('L')
    #image = np.array(image_pil, 'uint8')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #cv2.imshow("adding faces",image)
    label = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))
    print image_path
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        images.append(image[y: y+h, x: x+w])
        cv2.rectangle(image, (x,y), (x+w, y+w), (0,255,0),2)
        labels.append(label)
        #cv2.imshow("adding faces",image[y:y+h, x:x+w])
        #cv2.waitKey(2)
    #cv2.destroyAllWindows()

#cv2.destroyAllWindows()


recognizer.train(images, np.array(labels))

while True:
    ret,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors = 6,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )
    #faces = faceCascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+w), (0,255,0),2)
        result = cv2.face.MinDistancePredictCollector()
        recognizer.predict(gray[y: y+h, x:x+h], result, 0)
        predicted = result.getLabel()
        conf = result.getDist()
        if conf>40:
            cv2.putText(frame, names[predicted]+str(conf), (x,y), cv2.FONT_ITALIC, 0.9, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
