#!/usr/bin/env python
# coding: utf-8

# # Retrieve Dataset

# In[64]:


import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np;

raw_ds = tfds.load('deep_weeds', split="train", as_supervised=True)


# # Pre-process Dataset

# In[65]:


def negative_class_filter(x, y):
  return tf.not_equal(y, 8)

def rgb_to_grayscale_map(x, y):
  return (tf.reduce_mean(x, 2), y)

def downsampling(x, y):
  return (tf.image.resize(x, [64, 64]), y)

full_ds = raw_ds.filter(negative_class_filter) # remove instances from negative class
full_ds_size = full_ds.reduce(0, lambda x,_: x + 1).numpy()
full_ds = full_ds.shuffle(full_ds_size) # shuffle dataset
full_ds = full_ds.map(downsampling)
# full_ds = full_ds.map(rgb_to_grayscale_map) # convert to grayscale


# # Train Test Split

# In[66]:


train_size = int(0.7 * full_ds_size)
test_size = int(0.3 * full_ds_size)

# train_size = 1000
# test_size = 1000
used_size = train_size + test_size

used_ds = full_ds.take(used_size)

train_ds = used_ds.take(train_size)
test_ds = used_ds.skip(train_size)

train_ds_numpy = tfds.as_numpy(train_ds)
test_ds_numpy = tfds.as_numpy(test_ds)

print(f'Total number of all instances = {full_ds_size}')
print(f'Total number of used instances = {used_size}')
print(f'Total number of training instances = {train_size}')
print(f'Total number of testing instances = {test_size}')

# TODO: Reduce dimension
# TODO: Abstract function


# In[67]:


train_X = []
train_Y = []
for ex in train_ds_numpy:
  x = ex[0].flatten()
  y = ex[1]
  train_X.append(x)
  train_Y.append(y)
print(f'Sample train_X: {train_X[0]}')
print(f'Dimension of train_X: {len(train_X[0])}')
print(f'Sample train_Y: {train_Y[0]}')

test_X = []
test_Y = []
for ex in test_ds_numpy:
  x = ex[0].flatten()
  y = ex[1]
  test_X.append(x)
  test_Y.append(y)
print(f'Sample test_X: {test_X[0]}')
print(f'Dimension of test_X: {len(test_X[0])}')
print(f'Sample test_Y: {test_Y[0]}')


# # k-Fold Cross Validation

# In[68]:


def get_accuracy(predicted_Y, actual_Y):
  total_count = len(predicted_Y)
  correct_count = 0
  for i in range(total_count):
    if predicted_Y[i] == actual_Y[i]:
      correct_count += 1
  accuracy = correct_count / total_count
  return accuracy


# In[69]:


from sklearn.model_selection import KFold
from sklearn import tree

def run_decision_tree(train_X, train_Y, test_X, test_Y, depth):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(train_X, train_Y)
    predicted_Y = clf.predict(test_X)
    accuracy = get_accuracy(predicted_Y, test_Y)
    return accuracy

def kfold_cross_validation(k, train_X, train_Y, depth):
  print(f'Running {k}-fold cross validation for depth: {depth}')
  k = 5
  kf = KFold(n_splits=k)

  accuracy_scores = []

  for train_index, test_index in kf.split(train_X):
    cf_train_X = [train_X[index] for index in train_index]
    cf_train_Y = [train_Y[index] for index in train_index]
    cf_test_X = [train_X[index] for index in test_index]
    cf_test_Y = [train_Y[index] for index in test_index]
    accuracy = run_decision_tree(cf_train_X, cf_train_Y, cf_test_X, cf_test_Y, depth)
    accuracy_scores.append(accuracy)
    print(f'Obtained split accuracy of: {accuracy}')

  average_accuracy = sum(accuracy_scores) / k
  print(f'Completed {k}-fold cross validation for depth: {depth}')
  print(f'Obtained average accuracy of: {average_accuracy}')
  return average_accuracy


# In[70]:


def get_best_depth(depths):
  accuracy_scores = []
  k = 5
  print(f'Running {k}-fold cross validation for depths: {depths}')
  for depth in depths:
    average_accuracy = kfold_cross_validation(k, train_X, train_Y, depth)
    accuracy_scores.append(average_accuracy)

  index = accuracy_scores.index(max(accuracy_scores))
  print(f'Average scores for each depth: {accuracy_scores}')
  print(f'Best depth: {depths[index]}')
  return depths[index]

depths = [4, 6, 8, 10, 12, 16, 18, 20]
best_depth = get_best_depth(depths)


# # Test Model

# In[72]:


final_accuracy = run_decision_tree(train_X, train_Y, test_X, test_Y, best_depth)
print(f'Obtained final accuracy of {final_accuracy}')

