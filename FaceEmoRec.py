
from google.colab import drive
drive.mount('/content/drive')

!pip install keras_preprocessing
from keras.utils import to_categorical
from keras.models import Sequential
from keras_preprocessing.image import load_img
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import os
import numpy as np
import pandas as pd
!unzip /content/drive/MyDrive/ColabNotebooks/images.zip

TRAIN_DIR = "images/train"
TEST_DIR = "images/test"

def create_dataframe(dir):
    image_paths = []
    labels = []
    for lable in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,lable)):
            image_paths.append(os.path.join(dir,lable,imagename))
            labels.append(lable)
        print(lable,"completed")
    return image_paths,labels

train=pd.DataFrame()
train['image'],train['label']=create_dataframe(TRAIN_DIR)

test=pd.DataFrame()
test['image'],test['label']=create_dataframe(TEST_DIR)

from tqdm.notebook import tqdm
def extract_features(images):
    features = []
    for image in tqdm(images):
        img=load_img(image,grayscale=True)
        img=np.array(img)
        features.append(img)
    features=np.array(features)
    features=features.reshape(len(features),48,48,1)
    return features

train_features=extract_features(train['image'])

test_features = extract_features(test['image'])

x_train=train_features/255.0
x_test=test_features/255.0

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
le.fit(train['label'])

y_train =  le.transform(train['label'])
y_test = le.transform(test['label'])

y_train=to_categorical(y_train,num_classes=7)
y_test=to_categorical(y_test,num_classes=7)

model=Sequential()
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=x_train,y=y_train,batch_size=128,epochs=100,validation_data=(x_test,y_test))

mode_json= model.to_json()
with open("emotiondetection.json", "w") as json_file:
    json_file.write(mode_json)
model.save("emotiondetection.h5")

from keras.models import model_from_json

json_file=open("emotiondetection.json","r")
model_json=json_file.read()
json_file.close()
model=model_from_json(model_json)
model.load_weights("emotiondetection.h5")

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def ef(image):
    img=load_img(image,grayscale=True)
    feature=np.array(img)
    feature =feature.reshape(1,48,48,1)
    return feature/255.0

image ='images/test/surprise/10306.jpg'
print("original",image)
img =ef(image)
pred=model.predict(img)
pred_label = label[pred.argmax()]
print("predicted is ",pred_label)

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Classification Report:\n", classification_report(y_true_classes, y_pred_classes))
print("Accuracy:\n", accuracy_score(y_true_classes, y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(y_true_classes, y_pred_classes))
