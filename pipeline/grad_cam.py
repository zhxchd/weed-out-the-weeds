import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf


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
