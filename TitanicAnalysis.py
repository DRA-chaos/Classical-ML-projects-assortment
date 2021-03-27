#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd


# In[20]:


df=pd.read_csv("data.csv")


# In[21]:


df.head


# In[22]:


df.head(8)


# In[23]:


df.tail(3)


# In[24]:


df.columns


# In[25]:


df.rows


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.describe


# #Plotting Histogram of the age variable

# In[ ]:


df['Age'].plot.hist()


# In[ ]:


#PLotting a boxplot


# In[ ]:


df['Age'].plot.boxplot()


# In[26]:


df['Age'].plot.box()


# #Univariate Analysis for Categorical Variables

# In[27]:


#creating freequency table for ccategorical variable sex
df['Sex'].value_counts()


# In[28]:


#creating frequency percentage  table for ccategorical variable sex
df['Sex'].value_counts()/len(df['sex'])


# In[31]:



df['Sex'].value_counts()/len(df['Sex'])


# In[32]:


df['Sex'].value_counts().plot.bar()


# In[33]:


(df['Sex'].value_counts()/len(df['Sex'])).plot.bar()


# In[34]:


(df['Sex'].value_counts()/len(df['Sex'])).plot.box()


# In[35]:


df.head()


# In[36]:


df['Age'].plot.hist()


# In[37]:


df['Sex'].plot.hist()


# In[39]:


df['Sex'].value_counts()


# In[40]:


df.dtypes()


# In[41]:


df.dtypes


# 
# #Continuous continuous bivariate analysis
# df.plot.scatter('Age','Fare')

# In[43]:


df.corr()


# In[64]:


df['Fare'].corr(df['Age'])


# In[47]:


df.plot.scatter('Age','Fare')


# #Categorical Continuos Bivariate Analysis

# In[49]:


df.groupby('Sex')['Age'].mean().plot.bar()


# #Importing the scipy library for ttest

# In[50]:


from scipy.stats import ttest_ind


# In[56]:


df.groupby('Sex')['Age'].mean()


# In[57]:


males=df[df['Sex']=='male']


# In[58]:


females=df[df['Sex']=='female']


# In[59]:


ttest_ind(males['Age'],females['Age'],nan_policy='omit')


# In[60]:


ttest_ind(males['Age'],females['Age'])


# In[61]:


ttest_ind(males['Age'],females['Age'],nan_policy='omit')


# In[63]:


pd.crosstab(df['Sex'],df['Survived'])


# #Categorical Categorical analysis

# In[65]:


pd.crosstab(df['Sex'],df['Survived'])


# In[66]:


pd.crosstab(df['Sex'],df['Survived'])


# In[67]:


from scipy.stats import chi2_contingency


# In[68]:


chi2_contingency(pd.crosstab(df['Sex'],df['Survived']))


# In[69]:


df.describe()


# # Working with Missing Values

# In[70]:


df.describe()


# In[71]:


df.isnull()


# In[72]:


file.dropna()


# In[73]:


df.dropna()


# In[75]:


df.dropna().shape()


# In[77]:


df.dropna().isnull().sum()


# In[78]:


df.dropna().sum()


# In[79]:


df.dropna(how='all')


# In[81]:


df.dropna(how='all').shape


# In[82]:


df.dropna(axis=1)


# In[83]:


df.dropna(axis=1)df.dropna(axis=2)


# In[84]:


df.dropna(axis=1,how='all')


# In[85]:


df.dropna(axis=1,how='all').shape


# In[86]:


#for filling stuff and all

df.fillna(0)


# In[87]:


df['Age]'.fillna(0)


# In[88]:


df['Age'].fillna(0)


# In[89]:


df['Age'].fillna(df['Age'].mean())


# In[102]:


import matplotlib.pyplot as pt
import numpy as np


# In[91]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[92]:


df.head()


# #Univariate Outlier Detection

# In[93]:


#creating age boxplot


# In[94]:


df['Age'].plot.box()


# #bivariate outlier detection

# In[95]:


#Creating scatterplot for age and fare


# In[97]:


df.plot.scatter('Age','Fare')


# In[98]:


#Removing Outliers


# In[99]:


df=df[df['Fare']<300]


# In[100]:


df.plot.scatter('Age','Fare')


# In[101]:


#Replacing Outliers, replacing age with mean value of age


# In[103]:


df.loc[df['Age']>65,'Age']=np.mean(df['Age'])


# In[104]:


df.plot.scatter('Age','Fare')


# In[105]:


df['Age'].plot.box()


# In[106]:


df['Age'].plot.hist()


# In[107]:


np.log(df['Age']).plot.hist()


# In[108]:


np.sqrt(df['Age']).plot.hist()


# In[109]:


np.power(df['Age'],1/3).plot.hist()


# In[110]:


df['Age'].plot.hist()


# In[111]:


#Binning


# In[112]:


bins=[0,15,80]
group=['children','adult']


# In[113]:


df['type']=pd.cut(df['Age'],bins,labels=group)


# In[114]:


df.head


# In[115]:


df.head()


# In[ ]:





# In[116]:



df['type'].value_counts()


# In[ ]:




