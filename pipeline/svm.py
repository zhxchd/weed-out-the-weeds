# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from run_algo_with_kfold import kfold_cross_validation
from transfer_learning import init_conv_base, extract_features
from visualize import visualize, compare
from preprocess import retrieve_dataset, preprocess, to_np
from IPython import get_ipython

# %% [markdown]
# # Retrieve Dataset

# %%

(train_ds_raw, test_ds_raw), metadata = retrieve_dataset(should_shuffle_files=True)

train_ds = preprocess(train_ds_raw, {
    'is_filter_negative': True,
    'is_grayscale': False,
    'is_downsample64': False,
    'is_downsample128': False,
    'is_normalize': False,
})

test_ds = preprocess(test_ds_raw, {
    'is_filter_negative': True,
    'is_grayscale': False,
    'is_downsample64': False,
    'is_downsample128': False,
    'is_normalize': False,
})

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

visualize(X_train[0], y_train[0], metadata)
print(f'Length of X_train: {len(X_train)}')
print(f'Length of X_test: {len(X_test)}')

# %% [markdown]
# # Perform Transfer Learning

# %%
conv_base = init_conv_base(X_train[0])


# %%
train_features, train_labels = extract_features(
    conv_base, X_train, y_train)  # Agree with our small dataset size
test_features, test_labels = extract_features(conv_base, X_test, y_test)

# %% [markdown]
# ## Flatten To Feed into Different Algorithms

# %%
X_train_flatten = list(map(lambda x: x.flatten(), train_features))
y_train = train_labels
X_test_flatten = list(map(lambda x: x.flatten(), test_features))
y_test = test_labels

print(train_labels.shape)
# orginal feature is of dimension 196608
print("feature size now:", X_train[0].shape)


# %%
# For KNN, K Fold is just to look at prediction across various sets of data
# X_full = list(X_train_flatten)
# X_full.append(X_test_flatten)
# y_full = list(y_train)
# y_full.append(list(y_test))

# print(len(X_full))
# print(len(y_full))

# %% [markdown]
# # Test SVM Model with K-Fold Validation

# %%

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
k = 5
final_accuracies = []
for kernel in kernels:
    final_accuracy = kfold_cross_validation(
        k, X_train_flatten, y_train, 'knn', {'kernel': kernel})
    final_accuracies.append(final_accuracy)


# %%
print(final_accuracies)


# %%
