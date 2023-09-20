import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"
class Inception(tf.keras.Model):
    def __init__(self,c1,c2,c3,c4):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(c1, kernel_size=1,activation='relu', padding='same')
        self.conv2_1=tf.keras.layers.Conv2D(c2[0],kernel_size=1,activation='relu',padding='same')
        self.conv2_2=tf.keras.layers.Conv2D(c2[1],kernel_size=3,activation='relu',padding='same')
        self.conv3_1=tf.keras.layers.Conv2D(c3[0],kernel_size=1,activation='relu',padding='same')
        self.conv3_2=tf.keras.layers.Conv2D(c3[1],kernel_size=5,activation='relu',padding='same')
        self.pool4_1 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same',strides=1)
        self.conv4_2=tf.keras.layers.Conv2D(c4,kernel_size=1,activation='relu',padding='same')   
    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2_2(self.conv2_1(inputs))
        x3 = self.conv3_2(self.conv3_1(inputs))
        x4 = self.conv4_2(self.pool4_1(inputs))
        return tf.concat((x1, x2, x3, x4), axis=-1)
class GoogLeNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras. layers. Conv2D (filters=64, kernel_size=7, strides=2,activation='relu', padding = 'same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.conv2=tf.keras.layers.Conv2D(filters=64,kernel_size=1,activation='relu',padding='same')
        self.conv3=tf.keras.layers.Conv2D(filters=192,kernel_size=3,activation='relu',padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.inception1 = Inception(64, (96, 128), (16, 32), 32)
        self.inception2 = Inception(128, (128, 192), (32, 96), 64)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.inception3 = Inception(192, (96, 208), (16, 48), 64)
        self.inception4 = Inception(160, (112, 224), (24, 64), 64)
        self.inception5 = Inception(128, (128, 256), (24, 64), 64)
        self.inception6 = Inception(112, (144, 288), (32, 64), 64)
        self.inception7 = Inception(256, (160, 320), (32, 128), 128)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.inception8 = Inception(256, (160, 320), (32, 128), 128)
        self.inception9 = Inception(384, (192, 384), (48, 128), 128)
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.dense = tf.keras.layers.Dense(10)
    def call(self, inputs):
        x = self.pool1(self.conv1(inputs))
        x = self.pool2(self.conv3(self.conv2(x)))
        x = self.pool3(self.inception2(self.inception1(x)))
        x = self.pool4(self.inception7(self.inception6(self.inception5(self.inception4(self.inception3(x))))))
        x = self.dense(self.gap(self.inception9(self.inception8(x))))
        return x
net = GoogLeNet()
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net.layers:
    X = layer(X)
print(layer.name, 'output shape:\t', X.shape)
