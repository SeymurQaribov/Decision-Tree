#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Let's learn decision tree.
# A Decision Tree is a Flow Chart, and can help you make decisions based on previous experience.
import pandas as pd

df = pd.read_csv('C:/Users/Lenovo/Documents/winequality-red.csv')
df.head()


# In[24]:


# In the example, i will try to find the best wine features for the most quality wine.

# The feature columns(df[features]) are the columns that we try to predict from, 
# and the target column(df["quality"]) is the column with the values we try to predict.

features = ["fixed acidity","alcohol","fixed acidity","volatile acidity","residual sugar"]

X = df[features]

Y = df["quality"]

print(X)
print(Y)


# In[25]:


# Now let's import needed models

from  sklearn import tree

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

decision_tree = DecisionTreeClassifier()

dttree = decision_tree.fit(X,Y)

tree.plot_tree(dttree,feature_names = features )
plt.show() # I know ,this not so beatifull but i am in the begining of this machine learning. Let's imporve together.


# In[26]:


# Let's find the quality of wine by giving features of wine

print(dttree.predict([[10,10,10,0.1,5,]])) # The quality of wine is 6 with these features. 


# In[ ]:




