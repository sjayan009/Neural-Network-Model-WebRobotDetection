# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:41:26 2023

@author: jayan
"""

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split

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

print(features)
print(target_var)

X_train, X_test, y_train, y_test = train_test_split(features, target_var, test_size=0.2, random_state=0)


#Sequential Model - With Layers
model = Sequential()
model.add(Dense(60, input_dim=30, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(30, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=25, batch_size=10)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Accuracy: ", test_acc)

model.save('orange_nodropout', hist)

with open("my_model.pkcls", "wb") as file:
    pickle.dump(model, file)