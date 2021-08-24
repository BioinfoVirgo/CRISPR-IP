#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from codes.encoding import my_encode_on_off_dim
from codes import CRISPR_IP
import tensorflow as tf
import os
import sklearn


# In[2]:


seed = 123
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# In[3]:


# Incorporating reduced learning and early stopping for NN callback
eary_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.0001,
    patience=5, verbose=0, mode='auto')
callbacks = [eary_stopping]


# In[4]:


num_classes = 2
epochs = 500
batch_size = 4000
retrain=False
encoder_shape=(24,7)
seg_len, coding_dim = encoder_shape


# In[6]:


print('Encoding!!')
train_data = pd.read_csv('example_saved/example-train-data.csv')
test_data = pd.read_csv('example_saved/example-test-data.csv')
train_data_encodings = np.array(train_data.apply(lambda row: my_encode_on_off_dim(row['sgRNAs'], row['DNAs']), axis = 1).to_list())
train_labels = train_data.loc[:, 'labels'].values
test_data_encodings = np.array(test_data.apply(lambda row: my_encode_on_off_dim(row['sgRNAs'], row['DNAs']), axis = 1).to_list())
test_labels = test_data.loc[:, 'labels'].values
print('End of the encoding!!')


# In[7]:


xtrain, xtest, ytrain, ytest, inputshape = CRISPR_IP.transformIO(
    train_data_encodings, test_data_encodings, train_labels, test_labels, seg_len, coding_dim, num_classes)


# In[9]:


print('Training!!')
model = CRISPR_IP.crispr_ip(xtrain, ytrain, xtest, ytest, inputshape, num_classes, batch_size, epochs, callbacks, 
                            'example_saved/example', retrain)
print('End of the training!!')


# In[ ]:




