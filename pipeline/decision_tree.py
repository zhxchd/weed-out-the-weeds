#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from cv2 import cv2
from skimage import feature
from sklearn import neighbors


# # Retrieve Dataset

# In[ ]:


from preprocess import retrieve_dataset, preprocess, to_np

(train_ds_raw, test_ds_raw), metadata = retrieve_dataset(should_shuffle_files=True)

train_ds = preprocess(train_ds_raw, {
  'is_undersample_negative': False,
  'reduce_dataset_to': 0,
  'is_grayscale': False, 
  'is_downsample64': False,
  'is_downsample128': False, 
  'is_normalize': False,
})

test_ds = preprocess(test_ds_raw, {
  'is_undersample_negative': False,
  'reduce_dataset_to': 0,
  'is_grayscale': False, 
  'is_downsample64': False, 
  'is_downsample128': False, 
  'is_normalize': False,
})

def count_class(counts, batch):
    labels = batch[1]
    for i in range(9):
        cc = tf.cast(labels == i, tf.int32)
        counts[i] += tf.reduce_sum(cc)
    return counts

initial_state = dict((i, 0) for i in range(9))
counts = train_ds.reduce(initial_state=initial_state,
                         reduce_func=count_class)

print("Class breakdown for train dataset:")
print([(k, v.numpy()) for k, v in counts.items()])

initial_state = dict((i, 0) for i in range(9))
counts = test_ds.reduce(initial_state=initial_state,
                         reduce_func=count_class)

print("Class breakdown for test dataset:")
print([(k, v.numpy()) for k, v in counts.items()])


# # Extract X_train, Y_train, X_test, Y_test

# In[ ]:


train_ds_numpy = to_np(train_ds)
test_ds_numpy = to_np(test_ds)

X_train = [example for example, label in train_ds_numpy]
Y_train = [label for example, label in train_ds_numpy]

X_test = [example for example, label in test_ds_numpy]
Y_test = [label for example, label in test_ds_numpy]


# # Perform Transfer Learning

# In[ ]:


from transfer_learning import init_conv_base, extract_features
conv_base = init_conv_base(X_train[0])


# In[ ]:


train_features, train_labels = extract_features(conv_base, X_train, Y_train)
test_features, test_labels = extract_features(conv_base, X_test, Y_test)


# # Flatten To Fit Decision Tree

# In[ ]:


X_train_flatten = list(map(lambda x: x.flatten(), train_features))
Y_train = train_labels
X_test_flatten = list(map(lambda x: x.flatten(), test_features))
Y_test = test_labels

print(f'Number of training instances: {len(X_train_flatten)}')
print(f'Number of features: {len(X_train_flatten[0])}')


# # Find Best Depth Using K-fold Cross Validation

# In[ ]:


import importlib
import run_algo_with_kfold
importlib.reload(run_algo_with_kfold)
from run_algo_with_kfold import kfold_cross_validation, train_model, get_precision_scores

depths = [3, 4, 5, 6, 7, 8]
k = 5

accuracies = []

for depth in depths:
  accuracies.append(kfold_cross_validation(k, X_train_flatten, Y_train, 'decision_tree', {'depth': depth}))

highest_accuracy = 0
best_depth = 0

for idx, accuracy in enumerate(accuracies):
  if accuracy > highest_accuracy:
    best_depth = depths[idx]
    highest_accuracy = accuracy

print(f'Best depth: {best_depth}')


# # Train Final Model

# In[ ]:


model = train_model('decision_tree', {'depth': best_depth}, X_train_flatten, Y_train, True)

print(get_precision_scores(model, X_test_flatten, Y_test))

