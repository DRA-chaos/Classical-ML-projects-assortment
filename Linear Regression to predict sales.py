#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('train_cleaned.csv')


# In[3]:


data.head()


# In[4]:


#Segregating variables into dependent and Independent variables


# In[5]:


x=data.drop(['Item_Outlet_Sales'], axis=1)


# In[9]:


y=data['Item_Outlet_Sales']


# In[10]:


x.shape,y.shape


# In[11]:


#Splitting


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


train_x, test_x, train_y, test_y = train_test_split(x,y, random_state=56)


# # Implementing Linear Regression

# In[14]:


from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as mae


# In[15]:


#Creating instance of Linear Regression


# In[17]:


lr=LR()
lr.fit(train_x,train_y)


# In[18]:


train_predict=lr.predict(train_x)
k=mae(train_predict, train_y)


# In[19]:


print(' Training mean absolute error is   ', k)


# In[20]:


test_predict=lr.predict(test_x)
k=mae(test_predict, test_y)


# In[22]:


print(' Testing mean absolute error is   ', k)


# # Parameters of Linear Regression
# 

# In[23]:


#Parameters of LInear Regression
lr.coef_


# In[24]:


#Plotting the coefficients


# In[27]:


plt.figure(figsize=(8,6), dpi=120, facecolor='w', edgecolor='b')
x=range(len(train_x.columns))
y=lr.coef_
plt.bar(x,y)
plt.xlabel("variables")
plt.ylabel('Coefficients')
plt.title('Coefficient plot')


# # Checking assumptions of Linear Models

# In[30]:


#calculating residuals=actual-predicted
residuals=pd.DataFrame({'fitted values' : test_y, 'predicted values' : test_predict})
residuals['residuals']=residuals['fitted values']-residuals['predicted values']
residuals.head()


# Plotting Residual Curve

# In[33]:


plt.figure(figsize= (10,6), dpi=120, facecolor='w' , edgecolor='b')
f=range(0,2131)
k= [0 for i in  range (0,2131)]

plt.scatter(f, residuals.residuals[:], label='residuals')
plt.plot(f,k,color='red', label='regression line')
plt.xlabel('fitted points')
plt.ylabel('residuals')
plt.title('Residual plot')
plt.ylim(-4000,4000)
plt.legend()


# # Checking the distribution of Residuals

# In[34]:


# Histogram for distribution


# In[36]:


plt.figure(figsize=(10,6), dpi=120, facecolor='w', edgecolor='b')
plt.hist(residuals.residuals, bins=150)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Distribution of error terms')
plt.show()


# # QQ-Plot (is the data normally distributed ?)

# In[37]:


# importing the qq plot from statamodel


# In[39]:


from statsmodels.graphics.gofplots import qqplot


# In[40]:


#plotting qq plot


# In[41]:


fig,ax=plt.subplots(figsize=(5,5), dpi=120)
qqplot(residuals.residuals, line='s', ax=ax)
plt.ylabel('Residual Quantities')
plt.xlabel('Ideal scaled quantities')
plt.title('Checking distribution of residual errors')
plt.show()


# # Variance Inflation Factor

# In[42]:


# checking for multi-collinearity


# In[43]:


from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.tools.tools import add_constant


# In[44]:


#Calculating VIF for every column (only works for non-categorical)


# In[47]:


VIF=pd.Series([variance_inflation_factor(data.values, i) for i in range(data.shape[1])], index=data.columns)


# In[48]:


VIF


# # Model Interpretability

# In[49]:


#normalizing data is important


# In[51]:


lr=LR(normalize=True)


# In[52]:


#fitting the model


# In[53]:


lr.fit(train_x, train_y)


# In[54]:


#Predicting over train, test and calulating error


# In[55]:


train_predict=lr.predict(train_x)
k=mae(train_predict, train_y)
print (k)


# In[56]:


test_predict=lr.predict(test_x)
k=mae(test_predict, test_y)
print(k)


# In[57]:


plt.figure(figsize=(8,6), dpi=120, facecolor='w', edgecolor='b')
x=range(len(train_x.columns))
y=lr.coef_
plt.bar(x,y)
plt.xlabel("variables")
plt.ylabel('Coefficients')
plt.title('Normalized Coefficient plot')


# In[58]:


#Creating new data subsets


# In[59]:


x=data.drop(['Item_Outlet_Sales'], axis=1)
y=data['Item_Outlet_Sales']
x.shape, y.shape
Coefficients=pd.DataFrame({'Variable' : x.columns, 'coefficient' : lr.coef_})
Coefficients.head()


# In[60]:


sig_var=Coefficients[Coefficients.coefficient>0.5]


# In[61]:


subset=data[sig_var['Variable'].values]
subset.head()


# In[62]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y, random_state=56)
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as mae


# In[64]:


lr=LR(normalize=True)
lr.fit(train_x,train_y)


# In[ ]:




