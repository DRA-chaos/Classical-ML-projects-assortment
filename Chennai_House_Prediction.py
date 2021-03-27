#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd


# In[18]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Exploration and Manipulation

# In[19]:


df=pd.read_csv("chennai_house_price_prediction.csv")


# In[20]:


df.shape


# In[21]:


df.head()


# In[22]:


df.describe()


# In[23]:


df.describe(include='all')


# In[24]:


df.dtypes


# In[25]:


#Creating a new dataframe to summarize


# In[26]:


temp=pd.DataFrame(index=df.columns)
temp['data_type']=df.dtypes
temp['null_count']=df.isnull().sum()
temp['unique_count']=df.nunique() #to get the count of unique values


# In[27]:


temp


# # Univariate Analysis

# In[28]:


#Histogram
##target-variable

df['SALES_PRICE'].plot.hist(bins=50)
plt.xlabel('sales',fontsize=12)


# In[29]:


(df['SALES_PRICE'].loc[df['SALES_PRICE']<18000000]).plot.hist(bins=50)


# In[30]:


##Area of house in square feet


# In[31]:


df['INT_SQFT'].plot.hist(bins=50)
plt.xlabel('Area in square feet', fontsize=12)


# In[ ]:





# In[32]:


df['COMMIS'].plot.hist(bins=50)
plt.xlabel('Commission', fontsize=12)


# In[33]:


(df['COMMIS'].loc[df['COMMIS']<300000]).plot.hist(bins=50)


# # Value Counts

# In[34]:


#no of bedrooms


# In[35]:


df['N_BEDROOM'].value_counts()


# In[36]:


df['N_BEDROOM'].value_counts()/len(df)*100


# In[37]:


#BAr PLOT


# In[38]:


df['N_BATHROOM'].value_counts().plot(kind='bar')


# In[39]:


df['AREA'].value_counts().plot(kind='bar')


# In[40]:


df.isnull().sum()


# In[41]:


df.dropna(axis=0,how='any')


# In[42]:


df.dropna(axis=1,how='any')


# In[43]:


df['N_BEDROOM'].fillna(value=(df['N_BEDROOM'].mode()[0]),inplace=True)


# In[44]:


df.head()


# In[45]:


df.isnull().sum()


# In[46]:


df.loc[df['N_BATHROOM'].isnull()==True]


# In[47]:


for i in  range(0,len(df)):
    if pd.isnull(df['N_BATHROOM'][i])==True:
        if(df['N_BEDROOM'][i]==1.0):
            df['N_BATHROOM'][i]=1.0
        else:
            df['N_BATHROOM'][i]=2.0
                


# # 3.QS_OVERALL

# In[48]:


df[['QS_ROOMS','QS_BEDROOM','QS_BATHROOM','QS_OVERALL']].head()


# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


df=pd.read_csv("chennai_house_price_prediction.csv")


# In[51]:


df[['QS_ROOMS','QS_BEDROOM','QS_BATHROOM','QS_OVERALL']].head()


# In[52]:


temp=(df['QS_ROOMS'] + df['QS_BEDROOM']+df['QS_BATHROOM'])/3


# In[53]:


pd.concat([df['QS_ROOMS'] , df['QS_BEDROOM'],df['QS_BATHROOM'],df['QS_OVERALL'],temp],axis=1).head(10)


# In[54]:


df.loc[df['QS_OVERALL'].isnull()==True]


# In[55]:


df.loc[df['QS_OVERALL'].isnull()==True].shape


# In[56]:


def fill_na(x):
    return((x['QS_ROOMS']+x['QS_BATHROOM']+x['QS_BEDROOM'])/3)


# In[57]:


df['QS_OVERALL']=df.apply(lambda x:fill_na(x) if pd.isnull(x['QS_OVERALL'])else x['QS_OVERALL'],axis=1)


# In[ ]:


df.isnull().sum()
df=df.astype({'N_BEDROOM' : 'object' , 'N_ROOM' : 'object', 'N_BATHROOM' : 'object'})
df.isnull().sum()
# REPLACE CATEOGORIES


# # Bivariate Analysis

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df.columns()


# In[ ]:


df.columns


# In[ ]:


#Interior area and sale price
df.plot.scatter('INT_SQFT','SALES_PRICE')


# In[ ]:


fig,ax=plt.subplots()
colors={'Commercial':'red','House':'blue','Others':'green'}
ax.scatter(df['INT_SQFT'],df['SALES_PRICE'],c=df['BUILDTYPE'].apply(lambda x : colors[x]))


# In[ ]:


df=pd.read_csv("chennai_house_price_prediction.csv")


# In[ ]:


df.shape


# In[60]:


fig, ax=plt.subplots()
colors={'Commercial' : 'red' , 'House' : 'blue' , 'Others' : 'green'}


# In[66]:


fig, ax=plt.subplots()
colors={'Commercial' : 'red' , 'House' : 'blue' , 'Others' : 'green' , 'Other' : 'yellow'}
ax.scatter(df['INT_SQFT'], df['SALES_PRICE'], c=df['BUILDTYPE'].apply(lambda x : colors[x]))
plt.show()


# In[67]:


df.columns


# In[68]:


df['BUILDTYPE']


# # Replacing categories

# In[72]:


# first, we ought to view redundancies and spelling mistakes


# In[82]:


temp=['AREA', 'N_BEDROOM' , 'N_BATHROOM' , 'BUILDTYPE', 'PARK_FACIL']


# In[83]:


for i in temp:
    print ('***************** value count in', i , '********************')
    print(df[i].value_counts())
    print(" ")


# In[80]:


#To update or replace categories


# In[86]:


df['PARK_FACIL'].replace({'Noo': 'No'}, inplace=True)


# In[87]:


df['PARK_FACIL'].value_counts()


# In[89]:


df['BUILDTYPE'].replace({'Other' : 'Others', 'Comercial' : 'Commercial'}, inplace=True)


# In[90]:


df['BUILDTYPE'].value_counts()


# # Data Manipulation and Bivariate Analysis

# In[91]:


#Interior area and sale price
df.plot.scatter('INT_SQFT','SALES_PRICE')


# In[92]:


fig, ax=plt.subplots()
colors={'Commercial' : 'red' , 'House' : 'blue' , 'Others' : 'green'}
ax.scatter(df['INT_SQFT'], df['SALES_PRICE'], c=df['BUILDTYPE'].apply(lambda x : colors[x]))


# In[93]:


#2. SAles Price againnst no. of bedrooms and bathrooms


# In[99]:


# We make a pivot table for this
df[ 'N_BATHROOM']


# In[100]:


df.pivot_table(values='SALES_PRICE' , index='N_BEDROOM', columns='N_BATHROOM', aggfunc ='median')


# In[101]:


#3. QS_OVERALL and sales price


# In[102]:


df.plot.scatter('QS_OVERALL','SALES_PRICE')


# In[110]:


fig, axs=plt.subplot()
fig.set_figheight(10)
fig.set_figwidth(10)
axs.scatter(df['QS_BEDROOM'],df['SALES_PRICE'])
axs.set_title('QS_BEDROOM')


# In[111]:


df.plot.scatter('QS_OVERALL','SALES_PRICE')
fig, axs=plt.subplots(2,2)


# In[112]:


fig, axs=plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(10)


# In[114]:


fig, axs=plt.subplots(2,2)

fig.set_figheight(10)
fig.set_figwidth(10)

axs[0,0].scatter(df['QS_BEDROOM'],df['SALES_PRICE'])
axs[0,0].set_title('QS_BEDROOM')

axs[0,1].scatter(df['QS_BATHROOM'],df['SALES_PRICE'])
axs[0,1].set_title('QS_BATHROOM')

axs[1,0].scatter(df['QS_ROOMS'],df['SALES_PRICE'])
axs[1,0].set_title('QS_ROOMS')

axs[1,1].scatter(df['QS_OVERALL'],df['SALES_PRICE'])
axs[1,1].set_title('QS_OVERALL')


# In[115]:


#Create an axes instance 
ax=plt.figure().add_subplot(111)
ax.set_title('Quality store for houses')

#Create the boxplot
bp=ax.boxplot([df['QS_BEDROOM'],df['QS_ROOMS'],df['QS_BATHROOM'],df['QS_OVERALL']])


# In[125]:


#building type and sales price 

#Sales price based on building type, we make use of groupby function
df.groupby('BUILDTYPE').SALES_PRICE.median()

temp_df=df.loc[(df['BUILDTYPE']=='Commercial')& (df['AREA']=='Velachery')]
temp_df['SALES_PRICE'].plot.hist(bins=50)


# In[128]:





# In[120]:


df['AREA']


# In[130]:


#5/ SUrroundings and Locality

df.groupby(['BUILDTYPE','PARK_FACIL']).SALES_PRICE.median()


# In[133]:


rita=df.groupby(['BUILDTYPE','PARK_FACIL']).SALES_PRICE.median()
rita.plot(kind='bar', stacked= True)


# In[135]:


rita.plot(kind='bar')


# In[136]:


#6. AREA WISE PRICE FOR HOUSES
df.pivot_table(values='SALES_PRICE', index='AREA', aggfunc='median')


# In[137]:


temp_df=df.loc[(df['AREA']=='Karapakkam')]
temp_df['SALES_PRICE'].plot.hist(bins=50)


# In[138]:


#7 Distance from main road
df.plot.scatter('DIST_MAINROAD', 'SALES_PRICE')


# In[141]:


#8. Type of street around the house 
df.groupby(['STREET']).SALES_PRICE.median()


# # House sale price

# In[142]:


#Commission and sale price

df.plot.scatter('SALES_PRICE' , 'COMMIS')


# In[143]:


df[['SALES_PRICE','COMMIS']].corr()


# In[ ]:




