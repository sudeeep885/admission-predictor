import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


df = pd.read_csv("./dataset/Admission_Predict.csv")
df = df.drop('Serial No.', axis = 1)
df['Chance of Admit'] = [1 if each > 0.5 else 0 for each in df['Chance of Admit']]

x = df.drop('Chance of Admit', axis = 1)
y = df['Chance of Admit']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

lr = LogisticRegression(max_iter=500)
lr.fit(x_train, y_train)

pickle.dump(lr, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))