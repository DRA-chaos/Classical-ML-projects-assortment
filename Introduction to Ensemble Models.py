#!/usr/bin/env python
# coding: utf-8

# # Introduction to Ensemble Models

# In[1]:


#Max Voting
import pandas as pd


# In[2]:


import numpy as np


# In[3]:


data=pd.read_csv('data_cleaned.csv')


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


x=data.drop(['Survived'] , axis=1)
y=data['Survived']


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


train_x, valid_x,train_y,valid_y=train_test_split(x,y,random_state=101,stratify=y)


# In[9]:


from sklearn.linear_model import LogisticRegression


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[11]:


model1=LogisticRegression()
model1.fit(train_x,train_y)
pred1=model1.predict(valid_x)
pred1[:10],model1.score(valid_x,valid_y)


# In[12]:


model2=KNeighborsClassifier(n_neighbors=5)
model2.fit(train_x,train_y)
pred2=model2.predict(valid_x)
pred2[:10],model2.score(valid_x,valid_y)


# In[13]:


model3=DecisionTreeClassifier(max_depth=7)
model3.fit(train_x,train_y)
pred3=model3.predict(valid_x)
pred3[:10],model3.score(valid_x,valid_y)


# In[14]:


#Max Voting


# In[15]:


from statistics import mode
final_pred=np.array([])
for i in range(0,len(valid_x)):
    final_pred=np.append(final_pred,mode([pred1[i],pred2[i],pred3[i]]))


# In[16]:


from sklearn.metrics import accuracy_score
accuracy_score(valid_y,final_pred)
accuracy_score(valid_y,pred1),accuracy_score(valid_y,pred2),accuracy_score(valid_y,pred3)


# # Averaging (Regression)

# In[17]:


data=pd.read_csv('train_cleaned.csv')


# In[18]:


x=data.drop(['Item_Outlet_Sales'], axis=1)
y=data['Item_Outlet_Sales']


# In[19]:


from sklearn.model_selection import train_test_split
train_x, valid_x,train_y,valid_y=train_test_split(x,y,random_state=101,shuffle=False)


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


# In[21]:


model1=LinearRegression()
model1.fit(train_x,train_y)
pred1=model1.predict(valid_x)
pred1[:10],model1.score(valid_x,valid_y)


# In[22]:


model2=KNeighborsRegressor(n_neighbors=9)
model2.fit(train_x,train_y)
pred2=model2.predict(valid_x)
pred2[:10],model2.score(valid_x,valid_y)


# In[23]:


model3=DecisionTreeRegressor(max_depth=7)
model3.fit(train_x,train_y)
pred3=model3.predict(valid_x)
pred3[:10],model3.score(valid_x,valid_y)


# In[24]:


from statistics import mean 
final_pred = np.array([])
for i in range (0, len(valid_x)):
    final_pred=np.append(final_pred,mean([pred1[i],pred2[i],pred3[i]]))


# In[25]:


from sklearn.metrics import r2_score
r2_score(valid_y,final_pred)
r2_score(valid_y,pred1),r2_score(valid_y,pred2),r2_score(valid_y,pred3)


# In[26]:


r2_score(valid_y,final_pred)


# # Weighted Averaging

# In[27]:


from statistics import mean


# In[28]:


final_pred=np.array([])
for i in range (0,len(valid_x)):
    final_pred=np.append(final_pred,mean([pred1[i],pred1[i],pred2[i] , pred3[i], pred3[i]]))


# In[29]:


from sklearn.metrics import r2_score

r2_score(valid_y,final_pred)
# In[30]:


r2_score(valid_y,pred1),r2_score(valid_y,pred2),r2_score(valid_y,pred3)


# # Rank Averaging

# In[31]:


m1_score=model1.score(valid_x,valid_y)
m2_score=model2.score(valid_x,valid_y)
m3_score=model3.score(valid_x,valid_y)
m1_score,m2_score,m3_score


# In[32]:


model1=LinearRegression()
model1.fit(train_x,train_y)
pred1=model1.predict(valid_x)
pred1[:10],model1.score(valid_x,valid_y)


# In[33]:


model2=KNeighborsRegressor(n_neighbors=9)
model2.fit(train_x,train_y)
pred2=model2.predict(valid_x)
pred2[:10],model2.score(valid_x,valid_y)


# In[34]:


model3=DecisionTreeRegressor(max_depth=7)
model3.fit(train_x,train_y)
pred3=model3.predict(valid_x)
pred3[:10],model3.score(valid_x,valid_y)


# In[35]:


m1_score=model1.score(valid_x,valid_y)
m2_score=model2.score(valid_x,valid_y)
m3_score=model3.score(valid_x,valid_y)
m1_score,m2_score,m3_score


# In[36]:


index=[1,2,3]
valid_r2=[m1_score, m2_score,m3_score]
rank_eval=pd.DataFrame({'score' : valid_r2}, index=index_)
rank_eval


# In[ ]:





# In[ ]:





# In[ ]:




