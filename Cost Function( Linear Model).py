#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse


# # Creating Simple Data

# In[90]:


experience=[1.2,1.5,1.9,2.2,2.4,2.5,2.8,3.1,3.3,3.7,4.2,4.4]
salary=[1.7,2.4,2.3,3.1,3.7,4.2,4.4,6.1,5.4,5.7,6.4,6.2]


# In[91]:


data=pd.DataFrame({"salary" : salary, "experience" : experience})


# In[92]:


data.head()


# In[93]:


plt.scatter(data.experience, data.salary, color='red', label='data points')
plt.xlim(1,4.5)
plt.ylim(1,7)
plt.xlabel('experience')
plt.ylabel('salary')
plt.legend()


# # Starting the lines using small values of parameters

# In[94]:


# making lines for different values of beta, 0.1, 0.8, 1.5
data=pd.DataFrame({"salary" : salary, "experience" : experience})


# In[95]:


beta=1.5
# keeping the intercept constant
b=1.1
# to store predicted points

line1=[]
#generating predictions for each data point
for i in  range(len(data)):
    line1.append(data.experience[i]*beta + b)


# In[96]:


#plotting the line
plt.scatter(data.experience, data.salary, color='red')
plt.plot(data.experience, line1, color='black' , label='line')
plt.xlim(1,4.5)
plt.ylim(1,7)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
MSE=mse(data.experience, line1)
plt.title("Beta value" + str(beta) + "with MSE" + str(MSE))
MSE=mse(data.experience , line1)


# #plotting the line
# plt.scatter(data.experience, data.salary, color='red')
# plt.plot(data.experience, line1, color='black' , label='line')
# plt.xlim(1,4.5)
# plt.ylim(1,7)
# plt.xlabel('Experience')
# plt.ylabel('Salary')
# plt.legend()
# MSE=mse(data.experience, line1)
# plt.title("Beta value" + str(beta) + "with MSE" + str(MSE))
# MSE=mse(data.experience , line1)

# In[97]:


#plotting the line
plt.scatter(data.experience, data.salary, color='red')
plt.plot(data.experience, line1, color='black' , label='line')
plt.xlim(1,4.5)
plt.ylim(1,7)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
MSE=mse(data.experience, line1)
plt.title("Beta value" + str(beta) + "with MSE" + str(MSE))
MSE=mse(data.experience , line1)


# # Computing cost over a range of beta values

# In[98]:


#function to calculate error

def Error(Beta, data):
    b=1.1
    
    Salary=[]
    experience=data.experience
    for i in range(len(data.experience)):
        tmp=data.experience[i]*Beta + b
        salary.append(tmp)
    MSE=mse(experience,salary)
    return MSE


# In[101]:


# Range of slopes from 0 to 1.5 with increment of 0.01

Slope=[i/100 for i in range(0,150)]
Cost=[]
for i in Slope:
    cost=Error(Beta=i, data=data)
    Cost.append(cost)


# In[ ]:





# In[ ]:





# In[ ]:




