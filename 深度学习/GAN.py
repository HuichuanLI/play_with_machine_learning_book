# 生成数据集
# 目标圆是：圆心为（0,0），半径r=3 的圆，从上面采集10000个点作为训练数据集。
import json
import random
from matplotlib import pyplot as plt

num_max = 5000
r = 3
d = 2 * r
point = []
delta = d / num_max
x = -r
point_x = []
point_y = []
while len(point) < 2 * num_max:
    x = round(x, 8)
    y = (r ** 2 - x ** 2) ** 0.5
    y = round(y, 8)
    point.append([x, y])
    point.append([x, -y])
    point_x.append(x)
    point_y.append(y)
    point_x.append(x)
    point_y.append(-y)
    x += delta


# plt.scatter(point_x,point_y,linewidths=0.01)
# plt.show()


# 定义数据类
# 仿造Minist数据集的写法，实现一个data_generator类，实现其中的next_batch函数
class data_gen:
    def __init__(self):
        self.index = 0
        self.epoch = 0

    def next_batch(self, batchsize):
        rst = []
        if (self.index + 1) * batchsize > len(point):
            self.index = 0
            self.epoch += 1
        rst = point[self.index * batchsize:(self.index + 1) * batchsize]
        self.index += 1
        return rst


# 使用Tensorflow 实现一个简单的GAN
# GAN的一些设计：
# 判别器： MLP,3个隐藏层；每个输入包含两个点：来自实际的点x xx，以及生成器生成的假的点x ^ \hat x 
# x。隐藏层的神经元个数分别为：[16,32,20]，激活函数全部使用relu。最后输出层是一个sigmoid函数。

# 生成器：MLP,3个隐藏层；输入是一个随机噪声：z ∼ N ( 0 , 1 ) z\sim N(0,1)z∼N(0,1)；输出是一个假的点x ^ \hat x 
# x。 隐藏层神经元个数分别为：[20,32,32]；输出层具有两个神经元，分别表示横纵坐标。

# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import data_gen
import numpy as np
from matplotlib import pyplot as plt

dator = data_gen()

batch = 100
k = 5

realpoint_x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="realpoint")
noise_z = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="noise")

# 生成器
g_layers = [1, 20, 32, 32, 2]
# 第一层
Wg_1 = tf.Variable(tf.truncated_normal(shape=[g_layers[0], g_layers[1]], stddev=0.1), name="Wg1")
bg_1 = tf.Variable(tf.zeros(shape=[g_layers[1]]), "biasg1")
# 第二层
Wg_2 = tf.Variable(tf.truncated_normal(shape=[g_layers[1], g_layers[2]], stddev=0.1), name="Wg2")
bg_2 = tf.Variable(tf.zeros(shape=[g_layers[2]]), "biasg2")
# 第三层
Wg_3 = tf.Variable(tf.truncated_normal(shape=[g_layers[2], g_layers[3]], stddev=0.1), name="Wg3")
bg_3 = tf.Variable(tf.zeros(shape=[g_layers[3]]), "biasg3")
# 输出层
Wg_4 = tf.Variable(tf.truncated_normal(shape=[g_layers[3], g_layers[4]], stddev=0.1), name="Woutputg")
bg_4 = tf.Variable(tf.zeros(shape=[g_layers[4]]), "Woutputg")


def G(noise):
    print(type(noise))
    z = noise
    featureMapg1 = tf.nn.relu(tf.matmul(z, Wg_1) + bg_1)
    featureMapg2 = tf.nn.relu(tf.matmul(featureMapg1, Wg_2) + bg_2)
    featureMapg3 = tf.nn.relu(tf.matmul(featureMapg2, Wg_3) + bg_3)
    fake_x = 3 * tf.nn.tanh(tf.matmul(featureMapg3, Wg_4) + bg_4)
    return fake_x


# 鉴别器
d_layers = [2, 16, 30, 20, 1]
# 第一层的权值矩阵
Wd_1 = tf.Variable(tf.truncated_normal(shape=[d_layers[0], d_layers[1]], stddev=0.1), name="Wd1")
bd_1 = tf.Variable(tf.zeros(shape=[d_layers[1]]), "biasd1")

# 第二层的权值矩阵
Wd_2 = tf.Variable(tf.truncated_normal(shape=[d_layers[1], d_layers[2]], stddev=0.1), name="Wd2")
bd_2 = tf.Variable(tf.zeros(shape=[d_layers[2]]), "biasd2")

# 第三层的权值矩阵
Wd_3 = tf.Variable(tf.truncated_normal(shape=[d_layers[2], d_layers[3]], stddev=0.1), name="Wd3")
bd_3 = tf.Variable(tf.zeros(shape=[d_layers[3]]), "biasd3")

# 输出层的权值矩阵
Wd_4 = tf.Variable(tf.truncated_normal(shape=[d_layers[3], d_layers[4]], stddev=0.1), name="Woutputd")
bd_4 = tf.Variable(tf.zeros(shape=[d_layers[4]]), "Woutputd")


def D(input_x):
    x = input_x
    # x = tf.reshape(input_x,shape=[None,2])
    featureMapd1 = tf.nn.relu(tf.matmul(x, Wd_1) + bd_1)
    featureMapd2 = tf.nn.relu(tf.matmul(featureMapd1, Wd_2) + bd_2)
    featureMapd3 = tf.nn.relu(tf.matmul(featureMapd2, Wd_3) + bd_3)
    score = tf.nn.sigmoid(tf.matmul(featureMapd3, Wd_4) + bd_4)
    return score


fake_x = G(noise_z)

d_loss = -tf.reduce_mean(tf.log(D(realpoint_x)) + tf.log(1 - D(fake_x)))
g_loss = tf.reduce_mean(tf.log(1 - D(fake_x)))

# 使用var_list来指定只更新部分参数
d_learner = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss,
                                                                  var_list=[Wd_1, Wd_2, Wd_3, Wd_4, bd_1, bd_2, bd_3,
                                                                            bd_4])

g_learner = tf.train.AdamOptimizer(learning_rate=0.001).minimize(g_loss,
                                                                 var_list=[Wg_1, Wg_2, Wg_3, Wg_4, bg_1, bg_2, bg_3,
                                                                           bg_4])

if __name__ == '__main__':
    max_epoch = 1000
    step = 1
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        while dator.epoch < max_epoch:
            for _k in range(k):
                input_points = dator.next_batch(batch)
                noises = np.random.normal(size=[batch, 1])
                loss, learner = sess.run(fetches=[d_loss, d_learner],
                                         feed_dict={realpoint_x: input_points, noise_z: noises})
            input_points = dator.next_batch(batch)
            noises = np.random.normal(size=[batch, 1])
            loss, learner = sess.run(fetches=[g_loss, g_learner],
                                     feed_dict={realpoint_x: input_points, noise_z: noises})
            if step % 100 == 0:
                print({'step': step, 'loss': loss, 'epoch': dator.epoch})
            step += 1
        print("##############TEST#################")
        #     noises = np.random.normal(size=[batch,1])
        #     fake_point = sess.run(fetches=[fake_x],feed_dict = {noise_z:noises})[0]
        #     print(fake_point)
        #     plt.scatter(x=fake_point[:,0],y=fake_point[:,1],c='g',maker='*')
        # plt.show()
        noises = np.random.normal(size=[batch, 1])
        fake_point = sess.run(fetches=[fake_x], feed_dict={noise_z: noises})[0]
        print(fake_point)
        plt.scatter(x=fake_point[:, 0], y=fake_point[:, 1], c='g', marker='+')
        noises = np.random.normal(size=[batch, 1])
        fake_point = sess.run(fetches=[fake_x], feed_dict={noise_z: noises})[0]
        print(fake_point)
        plt.scatter(x=fake_point[:, 0], y=fake_point[:, 1], c='g', marker='*')
    plt.show()
