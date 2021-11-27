# Deep learning models

By Zhu Xiaochen, A0194512H

In this directory, we implemented deep learning models for the DeepWeeds dataset.

## Vanilla CNN model

In `deep_learning_vanilla_cnn.ipynb`, we experimented a vanilla CNN model with a few convolutional layers and max pooling layers. We observed that the model stops updating itself after a small number of iterations, which seems to be vanishing gradients.

## ResNet

To address vanishing gradients, we attempted residual networks (ResNet). ResNet is the state-of-the-art approach in image classification and it resolves vanishing gradients by propogating residuals and gradients via some shortcut connections, such that the gradients will not be as small.
 
### Simple ResNet models

`deep_learning_simple_resnet.ipynb` documents our experiments with a simple ResNet and its serious issue with overfitting. In the notebook, we attempted drop out layers, data augmentation and an even simpler model (because classical theories suggest that overfitting may be caused by over-parameterized models) to combat overfitting, but unfortunately none of them worked.

### ResNet-50 with pre-trained parameters

Model ML theories observe deep double descent, where after some threshold, more complicated models actually generalize better. Hence, we attempted ResNet-50 with pre-trained parameters on ImageNet as initialization weights. It turns out this approach has exceptional performance in our test trial (trained preliminarily on a small subset of data, documented in `deep_learning.ipynb`).

Hence, we proceeded to run full training data with this model, which is documented in the notebook `deep_learning_resnet50.ipynb`. We further evaluate the model via precision/recall/f1-score, as well as multi-class ROC curves. Eventually, we implemented `grad-CAM` to explain the model predictions. 