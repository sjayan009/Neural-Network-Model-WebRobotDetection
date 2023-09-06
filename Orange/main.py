# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:41:26 2023

@author: jayan
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

import pickle

df = pd.read_excel("simple_features.xlsx")
del df['ID']

# Preprocessing the Data (REMOVING null values)
df = df.dropna()

#Check to see if any null values remain
print(df.isnull().sum().sum())
#Test Passed -- ALL NULL VALUES REMOVED!

features = df.iloc[:, 0:30]
target_var = df.iloc[:, 30]

#Splitting Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(features, target_var, test_size=0.2, random_state=0)

#train_scores, test_scores = list(), list()

#Sequential Model - With Layers
model = Sequential()
model.add(Dense(60, input_dim=30, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(30, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#Compiling Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#Fitting Model into 'hist' variable
hist = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

#Evaluating on Train Dataset
#train_mp = (model.predict(X_train) >= 0.5).astype("int")
#train_acc_ov = accuracy_score(y_train, train_mp)
#train_scores.append(train_acc_ov)

#Evaluating on Test Dataset
#test_mp = (model.predict(X_test) >= 0.5).astype("int")
#test_acc_ov = accuracy_score(y_test, test_mp)
#test_scores.append(test_acc_ov)


plt.plot(hist.history['accuracy'], '-o', label='Train')
plt.plot(hist.history['val_accuracy'], '-o', label='Test')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Epoch vs Accuracy")
plt.legend()
plt.show()

#Saving Model
model.save('Overfitting.keras', hist)

with open("my_model.pkcls", "wb") as file:
    pickle.dump(model, file)