#import all the necessary libraries+
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder as le
#reading the dataset
df = pd.read_csv("cleaned lumpy dataset.csv")
print(df.head())
print(df['dominant_land_cover'].unique())
print(df.info())
print(df.describe())
#handling null values
print(df.isnull().sum()) #checking if we have null values
df["country"].fillna(df["country"].mode()[0],inplace=True) #replacing null values with its mode
print(df.isnull().sum())
#encoding categorical varibles
df["country"]=le().fit_transform(df["country"])
print(df["country"])
df["region"]=le().fit_transform(df["region"])
print(df["region"])
#deleting unwanted columns for the model
df.drop("Unnamed: 0", axis=1, inplace=True)
df.drop("X5_Bf_2010_Da", axis=1, inplace=True)
df.drop("X5_Ct_2010_Da", axis=1, inplace=True)
print(df.head())
#seperating dependent and independent variables
x = df.iloc[:,0:14]
x = pd.concat([x,df.iloc[:,15:]],axis=1)
print("XHead",x.shape)
y = df.loc[:,"lumpy"]
print(y.head())
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaled = sc.fit_transform(x)
#splitting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)
# -----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown="ignore"), [14,15])], remainder="passthrough")
x_transformed = ct.fit_transform(x)
#training the model with xgboost model
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)
# make a pickle file of our model
pickle.dump(model, open("model.pkl", "wb"))
joblib.dump(ct,"column")



