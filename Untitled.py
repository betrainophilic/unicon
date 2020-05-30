#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('titanic_train.csv')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df = pd.concat([df,sex,embark],axis=1)
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survived','Cabin','PassengerId'],axis=1),df['Survived'], test_size=0.30,random_state=101)
randomforest = RandomForestClassifier(n_estimators=100)
randomforest.fit(X_train, y_train)
pickle.dump(randomforest,open('titanic_model.sav','wb'))


# In[ ]:




