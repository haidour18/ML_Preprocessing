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

#Encoding Categorical data 

from sklearn.preprocessing import LabelEncoder

