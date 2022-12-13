#!/usr/bin/env python
# coding: utf-8

# 
# **<h1>Content Based Rcoemmender system for Auctions </h1>**
# 

# <h2>1- Import needed librairies and datasets</h2>

# In[28]:


import pandas as pd 
import numpy as np 

#Import dataset twice because the first one will be changed 
df2=pd.read_csv('C:/datasets/auctions.csv')
df3=pd.read_csv('C:/datasets/auctions.csv')


# <h2>2- Dataset overview</h2>

# In[29]:


df2.head(10)


# In[30]:



df2.info()


# <h2>3- Dataset cleaning</h2>

# In[31]:


df2=df2.drop(['__v'], axis=1)
df3=df3.drop(['__v'], axis=1)
df2 = df2.drop(df2[df2.active == False].index)
df3 = df3.drop(df3[df3.active == False].index)

df2 = df2.reset_index()
df3 = df3.reset_index()


# In[32]:


df3.shape


# <h2>4- Auction title based recommender</h2>

# In[33]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'....
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['title'] = df2['title'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['title'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[34]:


# print(tfidf_matrix)


# In[35]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[36]:


# Function that takes in auction auction_id as input and outputs most similar auctions
def get_recommendations_title(auction_id, cosine_sim=cosine_sim):
    # Get the index of the auction that matches the auction_id
    idx =df2.index[df2['_id'] == auction_id][0]
    
    # Get the pairwsie similarity scores of all auctions with that auction
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the auctions based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar auctions
    sim_scores = sim_scores[1:11]

    # Get the auction indices
    auction_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar auctions
    return df3[['title','description','category','user']].iloc[auction_indices]


# In[37]:


#get_recommendations_title("407f1f77bcf86cd79943901b")


# <h2>5- Auction metadata based recommender</h2>

# In[38]:



# from ast import literal_eval


## Parse the stringified features into their corresponding python objects
# features = ['keywords']
# for feature in features:
#     df2[feature] = df2[feature].apply(literal_eval)


# In[39]:


# # Returns the list top 3 elements or entire list; whichever is more.
# def get_list(x):
#     if isinstance(x, list):
#         names = [i['name'] for i in x]
#         #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
#         if len(names) > 3:
#             names = names[:3]
#         return names

#     #Return empty list in case of missing/malformed data
#     return []


# In[40]:


# # Define keywords features that are in a suitable form.


# features = [keywords']
# for feature in features:
#     df2[feature] = df2[feature].apply(get_list)


# In[41]:


df2[['category', 'title', 'description']].head(3)


# In[42]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    
    if isinstance(x, list):
        
        return [str.lower(i.replace(" ", "")) for i in x]
        
    else:
        #Check if director exists. If not, return empty string
       
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[43]:



# df3["keywords"] = df2["keywords"].apply(clean_data)


# In[44]:


# Apply clean_data function to your features.
features = ['category']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)


# In[45]:


def create_soup(x):
    #return ( x['category']+' '+x['title']+' '+x['keywords']+' '+x['user'])
    return ( x['category']+' '+x['title']+ ' '+x['description']+" "+x['user'])
df2['soup'] = df2.apply(create_soup, axis=1)
print(df2['soup'][0])


# In[46]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(df2['soup'])


# In[47]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[48]:


# Function that takes in auction auction_id as input and outputs most similar auctions
def get_recommendations_metadata(auction_id, cosine_sim=cosine_sim2):
    # Get the index of the auction that matches the auction_id
    idx =df2.index[df2['_id'] == auction_id][0]
    
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


# In[49]:


import pickle
pickle.dump(get_recommendations_metadata, open("model.pkl", "wb"))


# In[50]:


# model = pickle.load(open("./model.pkl", "rb"))
# my_personality = model('407f1f77bcf86cd79943901c')
# print('result: ', my_personality)


# 

# In[51]:


# get_recommendations_metadata('407f1f77bcf86cd79943901c')

