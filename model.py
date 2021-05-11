import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_excel("ev.xlsx")


X = df.drop("Fiyat",axis=1)
y = df["Fiyat"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


lr_model = LinearRegression()

lr_model.fit(X_train,y_train)

y_pred = lr_model.predict(X_test)

pickle.dump(lr_model,open("model.pkl","wb"))

model = pickle.load(open("model.pkl","rb"))










