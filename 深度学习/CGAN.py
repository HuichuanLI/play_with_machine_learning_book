#生成器模型的搭建
def Gnet(rand_x, y):
z_cond = tf.concat([rand_x, y], axis=1)  # 噪声的输入也要加上标签，
    w1 = tf.Variable(xavier_init([128 + 10, 128]))
    b1 = tf.Variable(tf.zeros([128]), dtype=tf.float32)
    y1 = tf.nn.relu(tf.matmul(z_cond, w1) + b1)
    w2 = tf.Variable(xavier_init([128, 784]))
    b2 = tf.Variable(tf.zeros([784]), dtype=tf.float32)
    y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
    # 待训练参数要一并返回
    params = [w1, b1, w2, b2]
    return y2, params
#判别器器模型的搭建
def Dnet(real_x, fack_x, y):
realx_cond = tf.concat([real_x, y], axis=1)  # 把原始样本和其标签一起放入
fackx_cond = tf.concat([fack_x, y], axis=1)  # 把生成样本和其伪造标签一起放入
    w1 = tf.Variable(xavier_init([784 + 10, 128]))
    b1 = tf.Variable(tf.zeros([128]), dtype=tf.float32)
    real_y1 = tf.nn.dropout(tf.nn.relu(tf.matmul(realx_cond, w1) + b1), 0.5)  # 不加dropout迭代到一定次数会挂掉
    fack_y1 = tf.nn.dropout(tf.nn.relu(tf.matmul(fackx_cond, w1) + b1), 0.5)
    w2 = tf.Variable(xavier_init([128, 1]))
    b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32)
    real_y2 = tf.nn.sigmoid(tf.matmul(real_y1, w2) + b2)
    fack_y2 = tf.nn.sigmoid(tf.matmul(fack_y1, w2) + b2)
    params = [w1, b1, w2, b2]
    return real_y2, fack_y2, params
