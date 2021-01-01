# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importer le dataset
dataset= pd.read_csv('Data.csv')
#Extraire les variables indépendantes 
X=dataset.iloc[:,:-1].values
#Extraire la variables dépendante
y=dataset.iloc[:,3].values
#Remplacer missing data 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding Categorical data les X

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
Labelencoder_X=LabelEncoder()
X[:,0]=Labelencoder_X.fit_transform(X[:,0])

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 0:]


#Encoding Categorical data Y 
Labelencoder_y=LabelEncoder()
y=Labelencoder_y.fit_transform(y)

#Spliting the dataset into training dataset and test dataset

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feauture scale 
from sklearn.preprocessing import StandardScaler 
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)



