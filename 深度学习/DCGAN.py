#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
def read_data():
data_dir = "data\mnist"
    #read training data
fd = open(os.path.join(data_dir,"train-images.idx3-ubyte"))
    loaded = np.fromfile(file = fd, dtype = np.uint8)
trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
fd = open(os.path.join(data_dir,"train-labels.idx1-ubyte"))
    loaded = np.fromfile(file = fd, dtype = np.uint8)
trainY = loaded[8:].reshape((60000)).astype(np.float)
    #read test data
fd = open(os.path.join(data_dir,"t10k-images.idx3-ubyte"))
    loaded = np.fromfile(file = fd, dtype = np.uint8)
testX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
fd = open(os.path.join(data_dir,"t10k-labels.idx1-ubyte"))
    loaded = np.fromfile(file = fd, dtype = np.uint8)
testY = loaded[8:].reshape((10000)).astype(np.float)
    # 将两个集合合并成70000大小的数据集
    X = np.concatenate((trainX, testX), axis = 0)
    y = np.concatenate((trainY, testY), axis = 0)
    print(X[:2])
    #set the random seed
    seed = 233
np.random.seed(seed)
np.random.shuffle(X)
np.random.seed(seed)
np.random.shuffle(y)
    return X/255, y


# In[2]:


#2.定义生成器
class Generator(object):
    """生成器"""
    def __init__(self, channels, init_conv_size):
        assert len(channels) > 1
self._channels = channels
        self._init_conv_size = init_conv_size
self._reuse = False
    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('generator', reuse=self._reuse):
            with tf.variable_scope('inputs'):
                fc = tf.layers.dense(inputs, self._channels[0] * self._init_conv_size * self._init_conv_size),conv0=tf.reshape(fc,[-1,self._init_conv_size,self._init_conv_size, self._channels[0]]),bn0 = tf.layers.batch_normalization(conv0, training=training),relu0 = tf.nn.relu(bn0)
deconv_input = relu0
            # deconvolutions * 4
            for i in range(1, len(self._channels)):
with_bn_relu = (i != len(self._channels) - 1)
deconv_inputs = conv2d_transpose(deconv_inputs,self._channels[i],
                                             'deconv-%d' % i,training,with_bn_relu)
img_inputs = deconv_inputs
            with tf.variable_scope('generate_imgs'):
                # imgs value scope: [-1, 1]
imgs = tf.tanh(img_inputs, name='imgaes')
self._reuse=True
        self.variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return imgs


# In[3]:


#3.定义生成器
class Discriminator(object):
    """判别器"""
    def __init__(self, channels):
self._channels = channels
self._reuse = False
    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
conv_inputs = inputs
        with tf.variable_scope('discriminator', reuse = self._reuse):
            for i in range(len(self._channels)):
conv_inputs = conv2d(conv_inputs,self._channels[i],'deconv-%d' % i,training)
fc_inputs = conv_inputs
            with tf.variable_scope('fc'):
                flatten = tf.layers.flatten(fc_inputs)
                logits = tf.layers.dense(flatten, 2, name="logits")
self._reuse = True
        self.variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return logits


# In[4]:


#4.定义损失函数
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))


# In[6]:


class DCGAN(object):
    """建立DCGAN模型"""
    def __init__(self, hps):
g_channels = hps.g_channels
d_channels = hps.d_channels
        self._batch_size = hps.batch_size
        self._init_conv_size = hps.init_conv_size
        self._batch_size = hps.batch_size
        self._z_dim = hps.z_dim
        self._img_size = hps.img_size
self._generator = Generator(g_channels, self._init_conv_size)
self._discriminator = Discriminator(d_channels)
    def build(self):
        self._z_placholder = tf.placeholder(tf.float32, (self._batch_size, self._z_dim))
        self._img_placeholder = tf.placeholder(tf.float32, (self._batch_size, self._img_size, self._img_size, 1))
generated_imgs = self._generator(self._z_placholder, training = True)
fake_img_logits = self._discriminator(generated_imgs, training = True)
real_img_logits = self._discriminator(self._img_placeholder, training = True)
loss_on_fake_to_real = tf.reduce_mean(
tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.ones([self._batch_size], dtype = tf.int64),logits = fake_img_logits))
loss_on_fake_to_fake = tf.reduce_mean(
tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.zeros([self._batch_size], dtype = tf.int64),logits = fake_img_logits))
loss_on_real_to_real = tf.reduce_mean(
tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.ones([self._batch_size], dtype = tf.int64),logits = real_img_logits))
tf.add_to_collection('g_losses', loss_on_fake_to_real)
tf.add_to_collection('d_losses', loss_on_fake_to_fake)
tf.add_to_collection('d_losses', loss_on_real_to_real)
        loss = {'g': tf.add_n(tf.get_collection('g_losses'), name = 'total_g_loss'),
              'd': tf.add_n(tf.get_collection('d_losses'), name = 'total_d_loss')}
        return (self._z_placholder, self._img_placeholder, generated_imgs, loss)
    def build_train(self, losses, learning_rate, beta1):
g_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1)
d_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1)
g_opt_op = g_opt.minimize(losses['g'], var_list = self._generator.variables)
d_opt_op = d_opt.minimize(losses['d'], var_list = self._discriminator.variables)
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name = 'train')
dcgan = DCGAN(hps)
z_placeholder, img_placeholder, generated_imgs, losses = dcgan.build()
train_op = dcgan.build_train(losses, hps.learning_rate, hps.beta1)


# In[ ]:


#6.训练模型
init_op = tf.global_variables_initializer()
train_steps = 10000
with tf.Session() as sess:
sess.run(init_op)
    for step in range(train_steps):
batch_img, batch_z = mnist_data.next_batch(hps.batch_size)
        fetches = [train_op, losses['g'], losses['d']]
should_sample = (step + 1) % 50 == 0
        if should_sample:
            fetches += [generated_imgs]
out_values = sess.run(fetches,feed_dict = {z_placeholder: batch_z,img_placeholder: batch_img})_, g_loss_val, d_loss_val = out_values[0:3]
        logging.info('step: %d, g_loss: %4.3f, d_loss: %4.3f' % (step, g_loss_val, d_loss_val))
        if should_sample:
gen_imgs_val = out_values[3]
gen_img_path = os.path.join(output_dir, '%05d-gen.jpg' % (step + 1))
gt_img_path = os.path.join(output_dir, '%05d-gt.jpg' % (step + 1))
gen_img = combine_and_show_imgs(gen_imgs_val, hps.img_size)
gt_img = combine_and_show_imgs(batch_img, hps.img_size)
            print(gen_img_path)
            print(gt_img_path)
gen_img.save(gen_img_path)
gt_img.save(gt_img_path)

