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


# # Train Decision Tree Model With K-Fold Cross Validation

# In[ ]:


import importlib
import run_algo_with_kfold
importlib.reload(run_algo_with_kfold)
from run_algo_with_kfold import kfold_cross_validation

depths = [5, 6, 7, 8, 9, 10]
k = 5
clfs_and_accuracies = []

for depth in depths:
  clf_and_accuracy = kfold_cross_validation(k, X_train_flatten, Y_train, 'decision_tree', {'depth': depth})
  clfs_and_accuracies.append(clf_and_accuracy)


# In[ ]:


print(clfs_and_accuracies)


# In[ ]:


import importlib
import run_algo_with_kfold
importlib.reload(run_algo_with_kfold)
from run_algo_with_kfold import get_precision_scores


# In[ ]:


for clf_and_accuracy in clfs_and_accuracies:
  (clf, accuracy) = clf_and_accuracy
  print(get_precision_scores(clf, X_test_flatten, Y_test))


# In[1]:


import importlib
import run_algo_with_kfold
importlib.reload(run_algo_with_kfold)
from run_algo_with_kfold import get_roc_auc_curve

fprs = []
tprs = []
roc_aucs = []
for clf_and_accuracy in clfs_and_accuracies:
  (fpr, tpr, roc_auc) = get_roc_auc_curve(clf, X_train_flatten, Y_train, X_test_flatten, Y_test, {'is_svm': False})
  fprs.append(fpr)
  tprs.append(tpr)
  roc_aucs.append(roc_auc)


# In[ ]:


import importlib
import run_algo_with_kfold
importlib.reload(run_algo_with_kfold)
from run_algo_with_kfold import visualize_roc_auc_curve
for depth, fpr, tpr, roc_auc in zip(depths, fprs, tprs, roc_aucs):
  title = f'ROC curve for depth = {str(depth)}'
  visualize_roc_auc_curve(title, fpr, tpr, roc_auc, len(np.unique(Y_test)))

