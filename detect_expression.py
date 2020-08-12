import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

cascadePath = 'haarcascade_frontalface_default.xml'
pathToModel = os.path.join('model', 'final_model_1.1251_0.5770_0.8951.h5')
classLabels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

faceCascade = cv2.CascadeClassifier(cascadePath)
videoCapture = cv2.VideoCapture(0)
model = load_model(pathToModel)

while True:
    _, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)

    if len(faces) != 0:
        # restricting to detecting only one face per frame
        (x, y, w, h) = faces[0]

        # crop out face from frame
        faceImg = gray[y:y + h, x:x + h]

        # preprocess image for model prediction
        faceImg = cv2.resize(faceImg, (48, 48), interpolation=cv2.INTER_AREA)
        faceImg = faceImg.astype('float32').reshape(1, 48, 48, 1)
        faceImg *= 1. / 255

        # get model predictions
        predictionProbabilities = model.predict(faceImg)[0]
        predictionInteger = np.argmax(predictionProbabilities, axis=-1)
        predictionClass = classLabels[predictionInteger]
        predictionClassProbability = str(predictionProbabilities.max())

        # render bounding box and prediction on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predictionClass, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, predictionClassProbability, (x + w - 50, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Expression Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
