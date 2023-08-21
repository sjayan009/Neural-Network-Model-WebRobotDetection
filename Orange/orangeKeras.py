# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:19:45 2023

@author: jayan
"""
import pandas as pd

from keras.models import load_model

model_w_dropout = load_model('orange_dropout.keras')
model_wo_dropout = load_model('orange_nodropout.keras')

prediction_df = pd.read_excel('random_test_data.xlsx')
del prediction_df['ID']

# Getting appropriate data
prediction_data_features = prediction_df.iloc[:, 1:31]
prediction_data_targetvar = prediction_df.iloc[:, 0]

predictions_1 = model_w_dropout.predict(prediction_data_features)
predictions_2 = model_wo_dropout.predict(prediction_data_features)

test_loss, test_acc = model_w_dropout.evaluate(prediction_data_features, prediction_data_targetvar)
test_loss_2, test_acc_2 = model_wo_dropout.evaluate(prediction_data_features, prediction_data_targetvar)

if test_acc_2 > test_acc:
    print("Model w/o Dropout is better model!")
else:
    print("Model w/ Dropout is better model!")
