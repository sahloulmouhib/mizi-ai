#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd 
import numpy as np 

#Import dataset twice because the first one will be changed 
df2=pd.read_csv('./auctions.csv')
df3=pd.read_csv('./auctions.csv')


# In[40]:


import pickle
count_matrix = pickle.load(open("./model.pkl", "rb"))


# In[41]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[42]:


df2=df2.drop(['__v'], axis=1)
df3=df3.drop(['__v'], axis=1)
df2 = df2.drop(df2[df2.active == False].index)
df3 = df3.drop(df3[df3.active == False].index)

df2 = df2.reset_index()
df3 = df3.reset_index()


# In[43]:


# Function that takes in auction auction_id as input and outputs most similar auctions
def get_recommendations_metadata(auction_id, cosine_sim=cosine_sim2):
    # Get the index of the auction that matches the auction_id
    idx =df3.index[df3['_id'] == auction_id][0]
    
    # Get the pairwsie similarity scores of all auctions with that auction
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the auctions based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar auctions
    sim_scores = sim_scores[1:11]

    # Get the auction indices
    auction_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar auctions
    return df3['_id'].iloc[auction_indices].to_json(orient = 'records')


# In[44]:


get_recommendations_metadata('407f1f77bcf86cd79943901c')


# In[ ]:




