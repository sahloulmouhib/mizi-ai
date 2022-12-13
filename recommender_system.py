#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import pymongo
from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://sahloulmouhib:sahloul1992@cluster0.qvcjt.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db=cluster['acutionAppDatabase']
collection=db['auctions']
df2 = pd.DataFrame(list(collection.find({})))



# In[44]:


import pickle
count_matrix = pickle.load(open("./model.pkl", "rb"))


# In[45]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[46]:


df2=df2.drop(['__v'], axis=1)

df2 = df2.drop(df2[df2.active == False].index)

df2 = df2.reset_index()

df3=df2.copy(deep=True)


# In[47]:


df3.head()


# In[48]:


print(type(str(df3['_id'][0])))


# In[49]:


from bson import ObjectId
# Function that takes in auction auction_id as input and outputs most similar auctions
def get_recommendations_metadata(auction_id, cosine_sim=cosine_sim2):
    try:
        # Get the index of the auction that matches the auction_id

        #print(df3.index[str(df3['_id']) == auction_id][0])
        idx =df3.index[df3['_id'] == ObjectId(auction_id)][0]
    
        # Get the pairwsie similarity scores of all auctions with that auction
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the auctions based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar auctions
        sim_scores = sim_scores[1:6]

        # Get the auction indices
        auction_indices = [i[0] for i in sim_scores]
    except:
        return []
    # total_rows['ColumnID'] = total_rows['ColumnID'].astype(str)
    # Return the top 10 most similar auctions
    return list(df3['_id'].astype(str).iloc[auction_indices])


# In[ ]:





# In[50]:


#get_recommendations_metadata('407f1f77bcf86cd79943901a')


# In[ ]:




