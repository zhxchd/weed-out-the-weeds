# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Import Libraries

# %%
get_ipython().run_line_magic('pip', 'install -r requirements.txt')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import pickle
from cv2 import cv2
from skimage import feature
from sklearn import neighbors

# %% [markdown]
# # Retrieve Dataset

# %%
from preprocess import retrieve_dataset, preprocess, to_np, count_class

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

# %% [markdown]
# # Extract X_train, Y_train, X_test, Y_test

# %%
train_ds_numpy = to_np(train_ds)
test_ds_numpy = to_np(test_ds)


# %%
X_train = [example for example, label in train_ds_numpy]
y_train = [label for example, label in train_ds_numpy]

X_test = [example for example, label in test_ds_numpy]
y_test = [label for example, label in test_ds_numpy]

# %% [markdown]
# # Visualize Preprocessed Dataset

# %%
from visualize import visualize, compare

visualize(X_train[0], y_train[0], metadata)
print(f'Length of X_train: {len(X_train)}')
print(f'Length of X_test: {len(X_test)}')

# %% [markdown]
# # Perform Transfer Learning

# %%
# Reference: https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751

from transfer_learning import init_conv_base, extract_features
conv_base = init_conv_base(X_train[0])


# %%
train_features, train_labels = extract_features(conv_base, X_train, y_train)  # Agree with our small dataset size
test_features, test_labels = extract_features(conv_base, X_test, y_test)

# %% [markdown]
# ## Flatten To Feed into Different Algorithms

# %%
X_train_flatten = list(map(lambda x: x.flatten(), train_features))
y_train = train_labels
X_test_flatten = list(map(lambda x: x.flatten(), test_features))
y_test = test_labels

print(train_labels.shape)
print("feature size now:", X_train[0].shape) # orginal feature is of dimension 196608

# %% [markdown]
# # Test SVM Model with K-Fold Validation

# %%
from run_algo_with_kfold import kfold_cross_validation

kernels = ['linear', 'rbf', 'poly', 'sigmoid']

k = 5
accuracies = [kfold_cross_validation(k, X_train_flatten, y_train, 'svm', {'kernel': kernel}) for kernel in kernels]


# %%
highest_index = np.argmax(accuracies)
best_kernel = kernels[highest_index]

print(f'Best kernel: {best_kernel}')


# %%
import importlib
import run_algo_with_kfold
importlib.reload(run_algo_with_kfold)
from run_algo_with_kfold import get_precision_scores, get_roc_auc_curve, visualize_roc_auc_curve, train_model


# %%
model = train_model('svm', {'kernel': best_kernel}, X_train_flatten, y_train, True)


# %%
print(get_precision_scores(model, X_test_flatten, y_test))


# %%
(fpr, tpr, roc_auc) = get_roc_auc_curve(model, X_train_flatten, y_train, X_test_flatten, y_test, {'is_svm': False})
print(fpr)
print(tpr)
print(roc_auc)