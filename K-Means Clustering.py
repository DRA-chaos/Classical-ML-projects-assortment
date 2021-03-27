#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd


# In[3]:


data=pd.read_csv('clustering.csv')


# In[4]:


data.head()


# In[5]:


data.shape()


# In[6]:


data.shape


# In[7]:


data.columns


# In[9]:


X=data[['LoanAmount' , 'ApplicantIncome']]


# In[10]:


#Visualize data points


# In[11]:


plt.scatter(X['ApplicantIncome'] , X['LoanAmount'], c='black')


# In[12]:


plt.scatter(X['ApplicantIncome'] , X['LoanAmount'], c='black')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In thousands)')
plt.show


# In[13]:


#Step1- Choose the no. of clusters(k) and select a random centroid for each cluster


# In[16]:


#no. of clusters 
k=3
#select random observations as centroid


# In[17]:


centroids=(X.sample(n=k))


# In[18]:


plt.scatter(X["ApplicantIncome"], X['LoanAmount'] , c='black')


# In[19]:


plt.scatter(centroids['ApplicantIncome'], centroids['LoanAmount'] , c='red')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (in thousannds)')
plt.show()


# In[20]:


#Step2: Assign all the points to the closest cluster centroid
#Step3: Recompute centroids of newly formed clusters


# In[21]:


#Step4: repeat steps 2 and 3


# In[31]:


diff=1
j=0
while(diff!=0):
    XD=X
    i=1
    for index1,row_c in centroids.iterrows():
        
        ED=[]
        for index2, row_d in XD.iterrows():
            
            d1=(row_c['ApplicantIncome']-row_d['ApplicantIncome'])**2
            d2=(row_c['LoanAmount']-row_d['LoanAmount'])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1
            
    C=[]
    for index, row in X.iterrows():
    
        min_dist=row[1]
        pos=1
        for i in range(k):
            if row[i+2]<min_dist:
                min_dist=row[i+2]
                pos=i+2
        c.append(pos)
    X['Cluster']=C
    centroids_new=X.groupby(['Cluster']).mean()[["LoanAmount","ApplicantIncome"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff=((centroids_new['LoanAmount']-centroids['LoanAmount']).sum() + (centroids_new['ApplicantIncome']-centroids['ApplicantIncome']))
        print(diff.sum())
    centroids=X.groupby(['Cluster']).mean()[['LoanAmount','ApplicantIncome']]

        


# In[ ]:




