#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk

from nltk import word_tokenize


# In[2]:


nltk.download('punkt')


# In[3]:


df = pd.read_csv("Desktop/bbc_text_cls.csv")


# In[4]:


df.head()


# In[5]:


idx = 0
word2idx = {}
tokenized_docs = []
for doc in df['text']:
    words = word_tokenize(doc.lower())
    doc_as_int = []
    for word in words:
        if word not in word2idx:
            word2idx[word] = idx
            idx =+ 1
        doc_as_int.append(word2idx[word])
    tokenized_docs.append(doc_as_int)          


# In[6]:


#reverse mapping
idx2word = {v:k for k,v in word2idx.items()}


# In[7]:


N = len(df['text'])


# In[8]:


V = len(word2idx)


# In[9]:


tf = np.zeros((N,V))


# In[10]:


for i, doc_as_int in enumerate(tokenized_docs):
    for j in doc_as_int:
        tf[i,j]+=1


# In[11]:


#compute IDF
document_freq =np.sum(tf>0, axis = 0)
idf = np.log(N/document_freq)


# In[12]:


#compute TF-IDF
tf_idf = tf * idf


# In[13]:


np.random.seed(10)


# In[14]:


i = np.random.choice(N)
row = df.iloc[i]
print ("label:",row['labels'])
print ("text",row['text'].split("\n",1)[0])
print ("Top 5 Terms:")

scores = tf_idf[i]
indices = (-scores).argsort()
for j in indices[:5]:
    print(idx2word[j])


# In[ ]:





# In[ ]:




