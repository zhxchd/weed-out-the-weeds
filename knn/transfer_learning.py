import os
import shutil
import numpy as np
from tensorflow.keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator


def init_conv_base(img):
    img_width, img_height = img.shape[0:2]
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(img_width, img_height, 3))  # 3 = number of channels in RGB pictures
    conv_base.summary()
    return conv_base

# Extract features


def extract_features(conv_base, x, y):
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 32
    sample_count = len(y)

    # Must be equal to the output of the convolutional base
    features = np.zeros(shape=(sample_count, 8, 8, 512))
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = datagen.flow(np.array(x), np.array(y), batch_size=batch_size)
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        # print(features_batch.shape)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
