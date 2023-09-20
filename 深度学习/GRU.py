# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)
tf.compat.v1.logging.set_verbosity(old_v)
batch_size = 100
time_step =28 # 时间步（每个时间步处理图像的一行）
data_length = 28 # 每个时间步输入数据的长度（这里就是图像的宽度）
learning_rate = 0.01
# 定义占位符
X_ = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.int32, [None, 10])
# dynamic_rnn的输入数据(batch_size, max_time, ...)
inputs = tf.reshape(X_, [-1, time_step, data_length])
# 验证集
validate_data = {X_: mnist.validation.images, Y_: mnist.validation.labels}
# 测试集
test_data = {X_: mnist.test.images, Y_: mnist.test.labels}
# 定义一个两层的GRU模型
gru_layers=rnn.MultiRNNCell([rnn.GRUCell(num_units=num) for num in [100, 100]], state_is_tuple=True)
outputs, h_ = tf.nn.dynamic_rnn(gru_layers, inputs, dtype=tf.float32)
output = tf.layers.dense(outputs[:, -1, :], 10) #获取GRU网络的最后输出状态
# 定义交叉熵损失函数和优化器
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y_, logits=output)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 计算准确率
accuracy=tf.metrics.accuracy(labels=tf.argmax(Y_,axis=1),predictions=tf.argmax(output, axis=1))[1]
## 初始化变量
sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
for step in range(3000):
    # 获取一个batch的训练数据
train_x, train_y = mnist.train.next_batch(batch_size)
    _, loss_ = sess.run([train_op, loss], {X_: train_x, Y_: train_y})
    # 在验证集上计算准确率
    if step % 100 == 0:
val_acc = sess.run(accuracy, feed_dict=validate_data)
        print('step:', step,'train loss: %.4f' % loss_, '| val accuracy: %.2f' % val_acc)
## 计算测试集史上的准确率
test_acc = sess.run(accuracy, feed_dict=test_data)
print('test loss: %.4f' % test_acc)
