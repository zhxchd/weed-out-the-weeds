#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from cv2 import cv2
from skimage import feature
from sklearn import neighbors


# # Retrieve Dataset

# In[2]:


from preprocess import retrieve_dataset, preprocess, to_np

(train_ds_raw, test_ds_raw), metadata = retrieve_dataset(should_shuffle_files=True)

train_ds = preprocess(train_ds_raw, {
  'is_undersample_negative': True,
  'reduce_dataset_to': 0,
  'is_grayscale': False, 
  'is_downsample64': False,
  'is_downsample128': False, 
  'is_normalize': False,
})

test_ds = preprocess(test_ds_raw, {
  'is_undersample_negative': True,
  'reduce_dataset_to': 0,
  'is_grayscale': False, 
  'is_downsample64': False, 
  'is_downsample128': False, 
  'is_normalize': False,
})


# # Extract X_train, Y_train, X_test, Y_test

# In[13]:


train_ds_numpy = to_np(train_ds)
test_ds_numpy = to_np(test_ds)

X_train = [example for example, label in train_ds_numpy]
Y_train = [label for example, label in train_ds_numpy]

X_test = [example for example, label in test_ds_numpy]
Y_test = [label for example, label in test_ds_numpy]


# # Perform Transfer Learning

# In[4]:


from transfer_learning import init_conv_base, extract_features
conv_base = init_conv_base(X_train[0])


# In[14]:


train_features, train_labels = extract_features(conv_base, X_train, Y_train)
test_features, test_labels = extract_features(conv_base, X_test, Y_test)


# # Flatten To Fit Decision Tree

# In[28]:


X_train_flatten = list(map(lambda x: x.flatten(), train_features))
Y_train = train_labels
X_test_flatten = list(map(lambda x: x.flatten(), test_features))
Y_test = test_labels


# # Train Decision Tree Model With K-Fold Cross Validation

# In[ ]:


import importlib
import run_algo_with_kfold
importlib.reload(run_algo_with_kfold)
from run_algo_with_kfold import kfold_cross_validation

depths = [5, 6, 7, 8, 9, 10]
final_accuracies = []
for n_neighbors in n_neighbors_arr:
  final_accuracy = kfold_cross_validation(k, X_train_flatten, y_train, 'decision_tree', {'depth': depths})
  final_accuracies.append(final_accuracy)

