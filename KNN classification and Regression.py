#!/usr/bin/env python
# coding: utf-8

# # KNN classification

# In[1]:


#Importing libraries


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[11]:


#Loading data


# In[12]:


data=pd.read_csv('data_cleaned.csv')


# In[13]:


data.shape


# In[14]:


data.head()


# # Segregating Variables : independent and Dependent variables

# In[15]:


x=data.drop(['Survived'], axis=1)
y=data['Survived']
x.shape, y.shape


# # Scaling the data(Using MinMax Scaler)

# In[16]:


##Importing the MinMax Scaler


# In[17]:


from sklearn.preprocessing import MinMaxScaler


# In[18]:


scaler=MinMaxScaler()


# In[19]:


x_scaled=scaler.fit_transform(x)


# In[20]:


x=pd.DataFrame(x_scaled,columns=x.columns)


# In[21]:


x.head()


# In[22]:


##Importing the train test split function


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


train_x, test_x, train_y, test_y=train_test_split(x,y,random_state=56, stratify=y)


# # Implementing the knn classifier

# In[25]:


#importing KNN classifier and metric F1 score


# In[26]:


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score


# In[27]:


#Creating instance of KNN


# In[28]:


clf=KNN(n_neighbors=5)


# In[29]:


#fitting the model


# In[30]:


clf.fit(train_x,train_y)


# In[31]:


test_predict=clf.predict(test_x)
k=f1_score(test_predict,test_y)
print('Test f1 score     ',k)


# # Elbow for Classifier

# In[33]:


def Elbow(K):
    #Initiating empty list
    test_error=[]
    #training model for every value of K
    for i in K:
        #instance of KNN
        clf=KNN( n_neighbors=i)
        clf.fit(train_x,train_y)
        #Appendinf f1 scores to empty list
        tmp=clf.predict(test_x)
        tmp=f1_score(tmp,test_y)
        error=1-tmp
        test_error.append(error)
        
    return test_error


# In[34]:


#Defining K range
k=range(6,20,2)


# In[35]:


test=Elbow(k)


# In[36]:


test


# In[37]:


#plotting the curves
plt.plot(k,test)
plt.xlabel('k Neighbors')
plt.ylabel('Test error')
plt.title('Elbow Curve for test')


# In[37]:


#creaing instance of KNN
clf=KNN(n_neighbors=12)
# fitting the model
clf.fit(train_x, train_y)
# Predicting over the train set and calculating F1
test_predict=clf.predict(test_x)
k=f1_score(test_predict,test_y)
print('Test f1 score  ', k)


# # KNN Regression

# In[38]:


data=pd.read_csv('train_cleaned.csv')


# In[39]:


data.shape


# In[40]:


data.head()


# In[41]:


#Segregating variables : Dependent and Independent


# In[42]:


x=data.drop(['Item_Outlet_Sales'], axis=1)


# In[43]:


y=data['Item_Outlet_Sales']


# In[44]:


x.shape
y.shape


# In[45]:


x.shape


# In[46]:


y.shape


# # Scaling the data using MinMaxScaler

# In[39]:


#Import
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x
                            )


# In[40]:


x=pd.DataFrame(x_scaled)


# In[41]:


x


# In[42]:


#Importing train test split


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


train_x, test_x, train_y, test_y = train_test_split(x,y, random_state=56)


# # Implementing KNN regressor

# In[45]:


#Importing KNN Regressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_squared_error as mse


# In[46]:


# Creating instance of KNN


# In[47]:


reg=KNN(n_neighbors=5)


# In[48]:


#Fitting the model
reg.fit(train_x,train_y)


# In[49]:


#Predicting over the train
test_predict=reg.predict(test_x)
k=mse(test_predict, test_y)
print('Test MSE   ', k)


# # Elbow for classifier

# In[56]:


def Elbow(K):
    test_mse=[]
    for i in K:
        reg=KNN(n_neighbors=i)
        reg.fit(train_x,train_y)
        tmp=reg.predict(test_x)
        tmp=mse(tmp,test_y)
        test_mse.append(tmp)
        
    return test_mse


# In[57]:


k=range(1,40)


# In[58]:


Elbow(k)


# In[59]:


test=Elbow(k)


# In[60]:


plt.plot(k,test)


# In[62]:


plt.plot(k,test)
plt.xlabel('k Neighbours')
plt.ylabel('Test Mean Squared Error')
plt.title('Elbow Curve for Test')


# In[63]:



reg=KNN(n_neighbors=18)

#Fitting the model
reg.fit(train_x,train_y)

#Predicting over the train
test_predict=reg.predict(test_x)
k=mse(test_predict, test_y)
print('Test MSE   ', k)


# In[ ]:




