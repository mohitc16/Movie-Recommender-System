
# coding: utf-8

# # importing packages
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# # reading data

# In[2]:


header = ['user_id','item_id','rating','timestamp']
df=pd.read_csv('ml-100k/u.data',sep='\t',names=header)


# In[3]:


df.head()


# In[4]:


#unique users
n_users=df['user_id'].nunique()
print(n_users)


# In[5]:


#unique movies
n_items=df['item_id'].nunique()
print(n_items)


# In[6]:


#train_test_split
from sklearn.cross_validation import train_test_split
train,test = train_test_split(df,test_size=0.25,random_state=0)


# In[7]:


header1=['user_id','age','gender','occupation','zip code']
user_details=pd.read_csv('ml-100k/u.user',delimiter='|',names=header1)


# In[8]:


user_details.head()


# In[9]:


header2=['movie_id','movie_title','release_date','video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy',
        'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
movie_details=pd.read_csv('ml-100k/u.item',sep='|',encoding='latin',names=header2)


# In[10]:


movie_details.drop('video_release_date',axis=1,inplace=True)


# In[11]:


movie_details.head()


# In[12]:


train_matrix=np.zeros((n_users,n_items))


# In[13]:


train_matrix.shape


# In[14]:


df.head()


# In[15]:


train.head()


# In[16]:


train.reset_index(drop=True,inplace=True)


# In[17]:


test.head()


# In[18]:


test.reset_index(drop=True,inplace=True)


# In[19]:


for i in range(0,train.shape[0]):
    user=train.loc[i]['user_id']-1
    item=train.loc[i]['item_id']-1
    rating=train.loc[i]['rating']
    train_matrix[user,item]=rating


# In[20]:


train_matrix


# # user similarity based collaborative filtering

# In[21]:


#cosine user-item similarity
from sklearn.metrics.pairwise import pairwise_distances
user_similarity=pairwise_distances(train_matrix,metric='cosine')


# In[22]:


user_similarity


# In[23]:


mean_user_rating=train_matrix.mean(axis=1)


# In[24]:


print(mean_user_rating)


# In[26]:


diff=train_matrix-mean_user_rating[:,np.newaxis]


# In[27]:


predictions=mean_user_rating[:,np.newaxis]+user_similarity.dot(diff)/np.array([np.abs(user_similarity).sum(axis=1)]).T


# In[28]:


predictions


# In[29]:


from sklearn import metrics
from math import sqrt


# In[30]:


test_matrix=np.zeros((n_users,n_items))
for i in range(0,test.shape[0]):
    user=test.loc[i]['user_id']-1
    item=test.loc[i]['item_id']-1
    rating=test.loc[i]['rating']
    test_matrix[user,item]=rating


# In[31]:


#RMSE
predict=predictions[test_matrix.nonzero()].flatten()
truth=test_matrix[test_matrix.nonzero()].flatten()
print("Error:")
print(sqrt(metrics.mean_squared_error(predict,truth)))


# # item similarity based collaborative filtering

# In[32]:


item_similarity=pairwise_distances(train_matrix.T,metric='cosine')


# In[33]:


item_similarity.shape


# In[34]:


item_similarity


# In[35]:


predictions1=train_matrix.dot(item_similarity)/np.array([np.abs(item_similarity).sum(axis=1)])


# In[36]:


predictions1.shape


# In[37]:


predictions1


# In[38]:


predict1=predictions1[test_matrix.nonzero()].flatten()
truth=test_matrix[test_matrix.nonzero()].flatten()


# In[39]:


print("Error:")
print(sqrt(metrics.mean_squared_error(predict1,truth)))

