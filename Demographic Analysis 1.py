#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('churn_prediction.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df['gender'].head()


# In[9]:


df['age'].plot.hist()


# In[10]:


df['age'].plot.box()


# In[11]:


df['gender'].value_counts()


# In[12]:


df.columns


# In[13]:


df['churn'].corr(df['days_since_last_transaction'])


# In[14]:



df['churn'].corr(df['current_balance'])


# In[15]:


df.groupby('churn')['age'].mean()


# In[16]:


df.groupby('churn')['dependents'].mean()


# In[17]:


# As it can be seen , churn rates are higher for those individuals who have a higher number of dependants which is quite intuitive 


# In[18]:


df.groupby('churn')['previous_month_end_balance'].mean()


# In[19]:


df.groupby('churn')['average_monthly_balance_prevQ'].mean()


# In[20]:


df.columns


# In[21]:


df.groupby('churn')['current_month_credit'].mean()


# In[22]:


df.isnull().sum()


# In[23]:


df['dependents'].mean()


# In[24]:


df['dependents'].fillna(df['dependents'].mean(), inplace=True)


# In[25]:


df.isnull().sum()


# In[26]:


df['gender'].head()


# In[27]:


df['occupation'].tail()


# In[28]:


df['occupation'].value_counts()


# In[29]:


df['occupation'].fillna('self_employed' , inplace=True)


# In[30]:


df.isnull().sum()


# In[31]:


df['city'].value_counts() , df['city'].mean()


# In[32]:


df['city'].fillna('796' , inplace=True)


# In[33]:


df.isnull().sum()


# In[34]:


df['days_since_last_transaction'].value_counts() , df['days_since_last_transaction'].mean(), df['days_since_last_transaction'].mode


# In[35]:


df['days_since_last_transaction'].fillna('69' , inplace=True)


# In[36]:


df.isnull().sum()


# In[37]:


df['age'].value_counts()


# In[38]:


bins=[0,15,60,91]
group=['children', 'adults' , 'senior citizens']


# In[39]:



df['type']=pd.cut(df['age'], bins , labels=group)


# In[40]:


df['type'].value_counts()


# In[41]:


df['Age_category']=type


# In[42]:


df.columns


# In[43]:


df['current_month_balance'].plot.hist(bins=50)


# In[44]:


df['churn'].value_counts().plot(kind='bar')


# In[45]:


df['gender'].mode()


# In[46]:


df['gender'].fillna('male', inplace=True) , df.isnull().sum()


# In[47]:


df.plot.scatter('previous_month_debit' , 'previous_month_end_balance')


# In[48]:


df[df['previous_month_end_balance']>5000000]


# In[49]:



ax=plt.figure().add_subplot()
bp=ax.boxplot([df['previous_month_credit'] , df['current_month_credit'] , df['current_month_debit'] , df['previous_month_debit']])


# In[50]:


df.groupby('occupation').current_month_credit.median()


# In[51]:


df.groupby('occupation').current_month_debit.median()


# In[52]:


df.groupby('occupation').current_month_balance.median()


# In[53]:


df.groupby('occupation').customer_nw_category.median()


# In[54]:


df['customer_nw_category'].value_counts()


# In[55]:


df.pivot_table(values='current_month_balance' , index='occupation' , aggfunc='median')


# In[56]:


df.pivot_table(values='current_month_credit' , index='occupation' , aggfunc='median')


# In[57]:


df.pivot_table(values='current_month_debit' , index='occupation' , aggfunc='median')


# In[58]:


df.plot.scatter('occupation' , 'current_month_balance')


# In[59]:


df.plot.scatter('occupation' , 'current_month_debit')


# In[60]:


# segregating variables


# In[61]:


df.columns


# In[62]:


df['churn'].head()


# In[63]:


x=df.drop(['churn'] , axis=1)
y=df['churn']


# In[64]:


x.shape , y.shape


# In[65]:


data.dtypes


# In[66]:


# I had made a small mistake in an above command which could not be revoked, so I am re-importing the csv file as another table 
#named data, but all the commands executed above which include the graphs are all accurate (because they were made before my mistake
# ALl the missing values have been mmodified or imputed


# In[67]:


data=pd.read_csv('churn_prediction.csv')


# In[68]:


data.describe()


# In[69]:


data.head()


# In[70]:


data.isnull().sum()


# In[71]:


data['days_since_last_transaction'].fillna('69' , inplace=True)
data['city'].fillna('796' , inplace=True)
data['occupation'].fillna('self_employed' , inplace=True)


# In[72]:


data.isnull().sum()


# In[73]:


data['dependents'].fillna(data['dependents'].mean(), inplace=True)


# In[74]:


bins=[0,15,60,91]
group=['children', 'adults' , 'senior citizens']
data['type']=pd.cut(data['age'], bins , labels=group)


# In[75]:


data['Age_category']=type


# In[76]:


data.describe()


# In[77]:


data['gender'].value_counts()


# In[78]:


data.isnull().sum()


# In[79]:


data['gender'].fillna('Male', inplace=True) , data.isnull().sum()


# In[80]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[81]:


data['gender'].value_counts()


# In[82]:


data['gender'].replace({'male' : 'Male'} , inplace=True )


# In[83]:


data['gender'].value_counts()


# In[84]:


pd.get_dummies(data['gender']).head()


# In[85]:


new=pd.get_dummies(data['gender'])


# In[86]:


new.head()


# In[87]:


df = pd.DataFrame(data)


# In[88]:


df.head()


# In[89]:


df['gender']=pd.get_dummies(data['gender'])


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[90]:


df['occupation']=pd.get_dummies(data['occupation'])
df.tail()


# In[91]:


df['type']=pd.get_dummies(data['type'])


# In[ ]:


df.head()


# In[92]:


df['occupation'].value_counts()


# In[93]:


df.dtypes


# In[94]:


df=pd.get_dummies(data)


# In[95]:


df.dtypes


# In[96]:


x=df.drop(['churn'], axis=1)
y=df['churn']


# In[97]:


x.shape, y.shape


# In[98]:


x_scaled=scaler.fit_transform(x)


# In[99]:


x=pd.DataFrame(x_scaled, columns=x.columns)


# In[100]:


x.head()


# In[101]:


# Importing the train test split function


# In[102]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y , random_state=56, stratify=y)


# # KNN Model

# In[103]:


#importing KNN classifier and metric F1 score


# In[104]:


from sklearn.neighbors import KNeighborsClassifier as KNC


# In[105]:


from sklearn.metrics import f1_score


# In[106]:


clf=KNC(n_neighbors=5)


# In[107]:


clf.fit(train_x, train_y)


# In[108]:


test_predict=clf.predict(test_x)
k=f1_score(test_predict, test_y)
print('Test F1 score  ' ,k)


# In[ ]:


#ELBOW for classifier


# In[ ]:


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


# In[ ]:


k=range(6,20,2)


# In[ ]:


test=elbow(k)


# In[ ]:




