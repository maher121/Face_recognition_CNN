# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 02:18:45 2021

@author: Maher
"""
# import math
import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
# import sklearn
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# import keras
# from keras.models import load_model
# from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16

# from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.utils import np_utils
# from keras.layers import Dense , Activation , Dropout ,Flatten
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.metrics import categorical_accuracy
# from keras.models import model_from_json
# from keras.callbacks import ModelCheckpoint
# from keras.optimizers import *
# from tensorflow.keras.layers import BatchNormalization
# import os

from keras.preprocessing import image
# from keras.applications.vgg16 import vgg16
# from keras.applications.vgg16 import preprocess_input



# print(os.listdir("../input/ck/"))
##---------------------------------------------------------------
model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
model.summary()
model_vgg16=Model(inputs=model.inputs,outputs=model.get_layer('block5_pool').output)

# Load and Prepare Dataset
data_path = 'Real And Fake Human Faces'
data_dir_list = os.listdir(data_path)
vgg16_feature_list=[]
data_y = []
i=0
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(224,224))
        img_data=image.img_to_array(input_img_resize)
        img_data=np.expand_dims(img_data, axis=0)
        img_data=preprocess_input(img_data)
        vgg16_feature=model_vgg16.predict(img_data)
        vgg16_feature_np=np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())
        data_y.append(i)
    i=i+1   
# %%
data_x = np.array(vgg16_feature_list)
data_y = np.array(data_y)

np.savetxt('data_x.csv', (data_x), delimiter=',')
np.savetxt('data_y.csv', (data_y), delimiter=',')

out  = data_y.reshape(1, -1).T
zz=np.concatenate([data_x,out],axis=1)

# %%
# data_x=pd.read_csv('data_x.csv')
# data_x.head()


#Shuffle the dataset
x,y = shuffle(data_x,data_y, random_state=2)
# Split the dataset
# Maher add 
X_train2, x_test, y_train2, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


##----------------  imbalance dataset ----------------
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_train, y_train = oversample.fit_resample(X_train2, y_train2)
# summarize class distribution
print(Counter(y_train))
##---------------- End imbalance dataset ----------------

# #%%
# # Machine Learning Classfier 
# # initializing all the model objects with default parameters

#%%##----------StackingClassifier and VotingClassifier
###An ensemble-learning meta-classifier for stacking.
#from sklearn import model_selection
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay


import numpy as np
import warnings
import xgboost as xgb

warnings.simplefilter('ignore')

clf1 = svm.SVC(kernel='poly', degree=8)
clf2 =  RandomForestClassifier(n_estimators=10, random_state=42)
clf3 = GaussianNB()
clf4 = KNeighborsClassifier(n_neighbors=10)
clf5 = xgb.XGBRFClassifier(use_label_encoder=False,eval_metric='logloss')

lr = LogisticRegression()


base_learners = [
                 ('rf_1', clf1),
                 ('rf_2', clf2) ,
                 ('rf_3',clf3),
                 ('rf_4', clf4) ,
                 ('rf_5',clf5)

                ]


# Initialize Stacking Classifier with the Meta Learner
sclf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
vclf = VotingClassifier(estimators=base_learners,voting='hard')

print ("--------------StackingClassifier and VotingClassifier -----------------")


for clff, label in zip([clf1, clf2, clf3,clf4, clf5, sclf,vclf], 
                      ['SVM', 
                       'Random Forest', 
                       'Naive Bayes',
                       'KNeighborsClassifier',
                       'xgboost',
                       'StackingClassifier',
                       'VotingClassifier'
                       ]):
    
    print ("----------------------- ",label,"-------------------------")
    clff.fit(X_train, y_train)
        # predicting the output on the test dataset
    pred_final = clff.predict(x_test)

   # print(confusion_matrix(y_test, pred_final))
    print(classification_report(y_test, pred_final))
    
    cm = confusion_matrix(y_test, pred_final)
    disp =ConfusionMatrixDisplay(confusion_matrix=cm)  
    disp.plot()
    plt.show()