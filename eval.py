import tensorflow as tf
import numpy as np
import cv2
import tensorflow as tf
import argparse
#import tensorflow.keras as keras
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.optimizers import Adam, Adagrad


eval_feature1 = np.load('val_features.npy')
feature2 = np.load('feature2.npy')
labels = np.load('labels.npy')


model = load_model('train_500.model')
val_loss, val_acc = model.evaluate({'A1':eval_feature1,'B1':feature2[12107:]},{'dense3':labels[12107:]})
print(val_loss, val_acc)