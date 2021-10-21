
import matplotlib.pyplot as plt

def visualize(image, label, metadata):
  get_label_name = metadata.features['label'].int2str
  _ = plt.imshow(image)
  _ = plt.title(get_label_name(label))

def compare(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
  