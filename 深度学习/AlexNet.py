import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from tensorflow import keras
class LRN(keras.layers.Layer):
    def __init__(self):
        super(LRN, self).__init__()
        self.depth_radius=2
        self.bias=1
        self.alpha=1e-4
        self.beta=0.75
    def call(self,x):
        return tf.nn.lrn(x,depth_radius=self.depth_radius,bias=self.bias,alpha=self.alpha,beta=self.beta)
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=96,kernel_size=(11,11),strides=4,activation='relu',padding='same',input_shape=(227,227,3)))
model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=2))
model.add(LRN())
model.add(keras.layers.Conv2D(filters=256,kernel_size=(5,5),strides=1,activation='relu',padding='same'))
model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=2))
model.add(LRN())
model.add(keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,activation='relu',padding='same'))
model.add(keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,activation='relu',padding='same'))
model.add(keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=1,activation='relu',padding='same'))
model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4096,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(4096,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1000,activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics = ["accuracy"])