import cv2
import numpy as np
from keras.models import model_from_json

json_file = open('emotiondetection.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('emotiondetection.h5')

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image, dtype='float32')
    feature = feature.reshape(1, 48, 48, 1)  # Model expects (1, 48, 48, 1)
    return feature / 255.0  

# Emotion labels
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    im = cv2.flip(im, 1)  # Horizontal flip

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        if roi_gray.shape[0] < 48 or roi_gray.shape[1] < 48:
            continue  # Skip if ROI is too small for the model
        resized_roi = cv2.resize(roi_gray, (48, 48))

        img = extract_features(resized_roi)

        pred = model.predict(img, verbose=0)
        prediction_label = labels[pred.argmax()]

        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(im, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
