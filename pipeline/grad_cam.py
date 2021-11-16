import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display

def sudo_gradcam(clf, pred, feature_map_size):
    coefs = clf.coef_[pred]
    result = np.zeros(int(coefs.size/feature_map_size))
    index = 0
    for coef in coefs:
        result[index] += coef
        index += 1
        if (index == int(coefs.size/feature_map_size)):
            index = 0
    return np.array(result)


def show_heat_map(clfs_and_accuracies, X_test, test_features, img_index=0):
    # take test case=img_index as example.
    imp_w = []
    X_test_flatten = list(map(lambda x: x.flatten(), test_features))
    for clf, accuracy in clfs_and_accuracies:
        for k in range(9):
            imp_w.append(sudo_gradcam(clf, k, 8*8))

        pred = int(clf.predict(X_test_flatten[img_index].reshape(1, -1))[0])
        heatmap = imp_w[pred][0] * test_features[img_index, :, :, 0]
        for i in range(1, 512):
            heatmap += imp_w[pred][i] * test_features[img_index, :, :, i]

        heatmap = tf.nn.relu(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        # print(pred, y_test[img_index])
        plt.matshow(heatmap)

        # Superimpose the heatmap on original image
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((256, 256))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        alpha = 0.4
        superimposed_img = jet_heatmap * alpha + X_test[img_index]
        superimposed_img = tf.keras.preprocessing.image.array_to_img(
            superimposed_img)
        plt.matshow(X_test[img_index])
        plt.matshow(superimposed_img)
    return imp_w


vgg = keras.applications.VGG16(weights='imagenet')

last_conv_layer_name = "block5_conv3"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        print(last_conv_layer_output.shape)
        print(preds.shape)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_heatmap (img_array):
    img_array = np.expand_dims(img_array, axis=0)
    return make_gradcam_heatmap(img_array, vgg, last_conv_layer_name)