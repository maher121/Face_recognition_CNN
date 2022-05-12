import os
from skimage import io
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import cv2

DatasetPath = []
for i in os.listdir("yalefaces"):
    DatasetPath.append(os.path.join("yalefaces", i))
imageData = []
imageLabels = []
for i in DatasetPath:
    imgRead = io.imread(i,as_gray=True)
    imageData.append(imgRead)
    
    labelRead = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
    imageLabels.append(labelRead)
#faceDetectClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDetectClassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceDetectClassifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")


imageDataFin = []
for i in imageData:
    facePoints = faceDetectClassifier.detectMultiScale(i)
    #facePoints = faceDetectClassifier.detectMultiScale(i, 1.1, 4)
    x,y = facePoints[0][:2]
    cropped = i[y: y + 150, x: x + 150]
    imageDataFin.append(cropped)
c = np.array(imageDataFin)
c.shape
X_train, X_test, y_train, y_test = train_test_split(np.array(imageDataFin),np.array(imageLabels), train_size=0.9, random_state = 20)
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train.shape
X_test.shape
nb_classes = 15
y_train = np.array(y_train) 
y_test = np.array(y_test)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
X_train = X_train.reshape(148, 150*150)
X_test = X_test.reshape(17, 150*150)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

#Defining the model()
model = Sequential()
model.add(Dense(512,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

#Training the model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy']);
history=model.fit(X_train, Y_train, batch_size=64, epochs=50, verbose=1, validation_data=(X_test, Y_test));
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Evaluating the performance
loss, accuracy = model.evaluate(X_test,Y_test, verbose=0);
print(loss)
print("Accuracy of model is: %",accuracy*100)
#predicted_classes = model.predict_classes(X_test)

predict_x=model.predict(X_test) 
predicted_classes=np.argmax(predict_x,axis=1)
correct_classified_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_classified_indices = np.nonzero(predicted_classes != y_test)[0]
print(correct_classified_indices)
print(incorrect_classified_indices)
