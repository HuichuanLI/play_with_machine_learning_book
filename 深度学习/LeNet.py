import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
(trainImage, trainLabel),(testImage, testLabel) = mnist.load_data()
trainImage = tf.reshape(trainImage,(60000,28,28,1))
testImage = tf.reshape(testImage,(10000,28,28,1))
net = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5),activation="relu",input_shape=(28,28,1),padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
    tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),activation="relu",padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5),activation="relu",padding="same"),
    #tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")    
])
net.summary()
net.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=["accuracy"])
net.fit(trainImage,trainLabel,epochs=5,validation_split=0.1)
testLoss, testAcc = net.evaluate(testImage,testLabel)
print(testAcc)
