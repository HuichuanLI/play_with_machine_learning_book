#!/usr/bin/env python
# coding: utf-8

# In[1]:


#配置环境
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import paddle 
import paddle.fluid as fluid 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
z_dim = 100 
batch_size = 128 
step_per_epoch = 60000 / batch_size


# In[2]:


# 定义生成器
def generator(z, name="G"): 
with fluid.unique_name.guard(name+'_'): 
fc1 = fluid.layers.fc(input = z, size = 1024) 
fc1 = fluid.layers.fc(fc1, size = 128 * 7 * 7)
fc1 = fluid.layers.batch_norm(fc1,act = 'tanh') 
fc1 = fluid.layers.reshape(fc1, shape=(-1, 128, 7, 7)) 
conv1 = fluid.layers.conv2d(fc1, num_filters = 4*64, filter_size=5, stride=1, padding=2, act='tanh') 
conv1 = fluid.layers.reshape(conv1, shape=(-1,64,14,14)) conv2 = fluid.layers.conv2d(conv1, num_filters = 4*32, filter_size=5, stride=1, padding=2, act='tanh') 
conv2 = fluid.layers.reshape(conv2, shape=(-1,32,28,28)) conv3 = fluid.layers.conv2d(conv2, num_filters = 1, filter_size=5, stride=1, padding=2,act='tanh') # 
conv3 = fluid.layers.reshape(conv3, shape=(-1,1,28,28)) print("conv3",conv3) 
return conv3


# In[3]:


# 定义判别器
def discriminator(image, name="D"):
    with fluid.unique_name.guard(name+'_'):
        conv1 = fluid.layers.conv2d(input=image, num_filters=32,
filter_size=6, stride=2,
                               padding=2)
        conv1_act = fluid.layers.leaky_relu(conv1)
        conv2 = fluid.layers.conv2d(conv1_act, num_filters=64, 
filter_size=6, stride=2, padding=2)
        conv2 = fluid.layers.batch_norm(conv2)
        conv2_act = fluid.layers.leaky_relu(conv2)
        fc1 = fluid.layers.reshape(conv2_act, shape=(-1,64*7*7))
        fc1 = fluid.layers.fc(fc1, size=512)
        fc1_bn = fluid.layers.batch_norm(fc1)
        fc1_act = fluid.layers.leaky_relu(fc1_bn)
        fc2 = fluid.layers.fc(fc1_act, size=1)
        print("fc2",fc2)
        return fc2


# In[ ]:


# 训练
def get_params(program, prefix):
all_params = program.global_block().all_parameters()
return [t.name for t in all_params if t.name.startswith(prefix)]
#优化generator
G_program = fluid.Program()
with fluid.program_guard(G_program):
    z = fluid.layers.data(name='z', shape=[z_dim,1,1])
    # 用生成器G生成样本图片
G_sample = generator(z)
infer_program = G_program.clone(for_test=True)
    # 用判别器D判别生成的样本
D_fake = discriminator(G_sample)
    ones = fluid.layers.fill_constant_batch_size_like(z, shape=[-1, 1], dtype='float32', value=1)
    # G损失
    # G Least square cost
G_loss = fluid.layers.mean(fluid.layers.square_error_cost(D_fake,ones))/2.
    # 获取G的参数
G_params = get_params(G_program, "G")   
    # 使用Adam优化器
G_optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
    # 训练G
G_optimizer.minimize(G_loss,parameter_list = G_params)
print(G_params)
# 优化discriminator
D_program = fluid.Program()
with fluid.program_guard(D_program):
    z = fluid.layers.data(name='z', shape=[z_dim,1,1])
    # 用生成器G生成样本图片
G_sample = generator(z)
    real = fluid.layers.data(name='img', shape=[1, 28, 28])
    # 用判别器D判别真实的样本
D_real = discriminator(real)
    # 用判别器D判别生成的样本
D_fake = discriminator(G_sample)
    # D损失
    print("D_real",D_real)
    print("D_fake",D_fake)
    # D Least square cost
    D_loss=fluid.layers.reduce_mean(fluid.layers.square(D_real-1.))/2.+ fluid.layers.reduce_mean(fluid.layers.square(D_fake))/2.
    print("D_loss",D_loss)
    # 获取D的参数列表
D_params = get_params(D_program, "D")
    # 使用Adam优化
D_optimizer = fluid.optimizer.Adam(learning_rate=0.0001)
D_optimizer.minimize(D_loss, parameter_list = D_params)
print(D_params)
# MNIST数据集，不使用label
def mnist_reader(reader):
    def r():
        for img, label in reader():
            yield img.reshape(1, 28, 28)
return r
#批处理
mnist_generator = paddle.batch(
paddle.reader.shuffle(mnist_reader(paddle.dataset.mnist.train()),1024),batch_size=batch_size)
z_generator = paddle.batch(z_reader, batch_size=batch_size)()

