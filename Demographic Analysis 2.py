#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('churn_prediction.csv')


# In[3]:


data.isnull().sum()


# In[4]:


bins=[0,15,60,91]
group=['children', 'adults' , 'senior citizens']
data['type']=pd.cut(data['age'], bins , labels=group)


# In[5]:


data['Age_category']=type


# In[6]:


data['days_since_last_transaction'].fillna('69' , inplace=True)
data['city'].fillna('796' , inplace=True)
data['occupation'].fillna('self_employed' , inplace=True)


# In[7]:


data['dependents'].fillna(data['dependents'].mean(), inplace=True)


# In[8]:


data['gender'].value_counts()


# In[9]:


data['gender'].fillna('Male', inplace=True) , data.isnull().sum()


# In[10]:


data.isnull().sum()


# In[11]:


data['gender'].value_counts()


# In[12]:


new=pd.get_dummies(data['gender'])


# In[13]:


df = pd.DataFrame(data)


# In[14]:


df['gender']=pd.get_dummies(data['gender'])


# In[15]:


df['occupation']=pd.get_dummies(data['occupation'])
df.tail()


# In[16]:


df['type']=pd.get_dummies(data['type'])


# In[17]:


df['occupation'].value_counts()


# In[18]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[19]:


df=pd.get_dummies(data)


# In[20]:


x=df.drop(['churn'], axis=1)
y=df['churn']


# In[21]:


x.shape, y.shape


# In[22]:


x_scaled=scaler.fit_transform(x)


# In[23]:


x=pd.DataFrame(x_scaled, columns=x.columns)


# In[24]:


x.head()


# In[25]:


# Importing the train test split function
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y , random_state=56, stratify=y)


# # KNN Model

# In[26]:


#importing KNN classifier and metric F1 score


# In[27]:


from sklearn.neighbors import KNeighborsClassifier as KNC


# In[28]:


from sklearn.metrics import f1_score


# In[29]:


clf=KNC(n_neighbors=5)


# In[30]:


clf.fit(train_x, train_y)


# In[31]:


test_predict=clf.predict(test_x)
k=f1_score(test_predict, test_y)
print('Test F1 score  ' ,k)


# In[32]:


#ELBOW for classifier


# In[33]:


def elbow(K):
    test_error=[]
    for i in K:
        clf=KNC(n_neighbors=i)
        clf.fit(train_x, train_y)
        tmp=clf.predict(test_x)
        tmp=f1_score(tmp, test_y)
        error=1-tmp
        test_error.append(error)
    return test_error


# In[36]:


k=range(6,10,2)


# In[37]:


test=elbow(k)


# In[38]:


plt.plot(k,test)
plt.xlabel('K Neighbors')
plt.ylabel('Test error')
plt.title('Elbow curve for test')


# In[39]:


# we take 5 to be the kNearest neighbors value


# In[41]:


y=df['churn']
x=df.drop(['churn'] , axis=1)


# In[43]:


train_y.value_counts(normalize=True)

test_y.value_counts(normalize=True)
# In[46]:


test_y.value_counts(normalize=True)


# In[47]:


from sklearn.tree import DecisionTreeClassifier


# In[48]:


dt_model=DecisionTreeClassifier(random_state=10)


# In[49]:


dt_model.fit(train_x, train_y)


# In[50]:


dt_model.score(train_x,train_y)


# In[51]:


dt_model.score(test_x, test_y)


# In[52]:


dt_model.predict(test_x)


# In[53]:


dt_model.predict_proba(test_x)


# In[54]:


y_pred=dt_model.predict_proba(test_x)[:,1]


# In[55]:


new_y=[]
for i in range(len(y_pred)):
    if y_pred[i]<0.6:
        new_y.append(0)
    else:
        new_y.append(1)


# In[56]:


from sklearn.metrics import accuracy_score


# In[57]:


accuracy_score(test_y,new_y)


# In[58]:


train_accuracy=[]
test_accuracy=[]
for depth in range(1,10):
    dt_model=DecisionTreeClassifier(max_depth=depth, random_state=10)
    dt_model.fit(train_x, train_y)
    train_accuracy.append(dt_model.score(train_x, train_y))
    test_accuracy.append(dt_model.score(test_x, test_y))


# In[59]:


frame=pd.DataFrame({'max_depth' : range(1,10) , 'train_acc' : train_accuracy , 'test_acc' : test_accuracy})
frame.head()


# In[60]:


plt.figure(figsize=(12,6))
plt.plot(frame['max_depth'] , frame['train_acc'] , marker='o')
plt.plot(frame['max_depth'] , frame['test_acc'] , marker='o')
plt.xlabel('depth of tree')
plt.ylabel('performance')
plt.legend()


# In[62]:


# we consider the max depth to be 5 from the above graph
dt_model=DecisionTreeClassifier(max_depth=5, max_leaf_nodes=25 , random_state=10)
dt_model.fit(train_x, train_y)
dt_model.score(train_x, train_y)
dt_model.score(test_x, test_y)


# In[63]:


from sklearn import tree


# In[65]:


decision_tree=tree.export_graphviz(dt_model, out_file='tree.dot', feature_names=train_x.columns, max_depth=2,filled=True)


# In[66]:


get_ipython().system('dot -Tpng tree.dot -o tree.png')


# In[67]:


image=plt.imread('tree.png')
plt.figure(figsize=(15,15))
plt.imshow(image)


# In[ ]:




