from operator import itemgetter
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds


def retrieve_dataset(should_shuffle_files: bool):
    (train_ds_raw, test_ds_raw), metadata = tfds.load(
        'deep_weeds',
        split=['train[:80%]', 'train[80%:]'],
        shuffle_files=should_shuffle_files,
        with_info=True,
        as_supervised=True,
    )
    return (train_ds_raw, test_ds_raw), metadata


def filter_negative(x, y):
    return tf.not_equal(y, 8)


def grayscale(x, y):
    return (tf.reduce_mean(x, 2), y)


def downsample64(x, y):
    return (tf.image.resize(x, [64, 64]), y)


def downsample128(x, y):
    return (tf.image.resize(x, [128, 128]), y)


def normalize(x, y):
    return (np.rint(255.0 * (x - np.min(x)) /
                    (np.max(x) - np.min(x).astype(x.dtype))), y)


def preprocess(dataset, options):
    (is_filter_negative,
     is_grayscale,
     is_downsample64,
     is_downsample128,
     is_normalize) = itemgetter(
        'is_filter_negative',
        'is_grayscale',
        'is_downsample64',
        'is_downsample128',
        'is_normalize'
    )(options)

    preprocessed_dataset = dataset

    if is_filter_negative is True:
        preprocessed_dataset = preprocessed_dataset.filter(filter_negative)

    if is_grayscale is True:
        preprocessed_dataset = preprocessed_dataset.map(grayscale)

    if is_downsample64 is True:
        preprocessed_dataset = preprocessed_dataset.map(downsample64)

    if is_downsample128 is True:
        preprocessed_dataset = preprocessed_dataset.map(downsample128)

    if is_normalize is True:
        preprocessed_dataset = preprocessed_dataset.map(normalize)

    return preprocessed_dataset


def to_np(dataset):
    return [(example.numpy(), label.numpy()) for example, label in dataset]
