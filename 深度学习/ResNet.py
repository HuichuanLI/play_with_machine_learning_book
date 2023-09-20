number of training examples = 1080
number of test examples = 120
X_train shape: (1080, 64, 64, 3)
Y_train shape: (1080, 6)
X_test shape: (120, 64, 64, 3)
Y_test shape: (120, 6)
x train max,0.956; x train min,0.015
x test max,0.94; x test min,0.011
def convolutional_block(self, X_input, kernel_size, in_filter,out_filters, stage, block, training, stride=2):
    # defining name basis
block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters
x_shortcut = X_input
        #first
        W_conv1 = self.weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)
        #second
        W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)
        #third
        W_conv3 = self.weight_variable([1,1, f2,f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        #shortcut path
W_shortcut = self.weight_variable([1, 1, in_filter, f3])
x_shortcut=tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')
        #final
        add = tf.add(x_shortcut, X)
add_result = tf.nn.relu(add)
    return add_result
def convolutional_block(self, X_input, kernel_size, in_filter,out_filters, stage, block, training, stride=2):
    # defining name basis
block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters
x_shortcut = X_input
        #first
        W_conv1 = self.weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)
        #second
        W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)
        #third
        W_conv3 = self.weight_variable([1,1, f2,f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        #shortcut path
W_shortcut = self.weight_variable([1, 1, in_filter, f3])
        x_shortcut=tf.nn.conv2d(x_shortcut,W_shortcut,strides=[1,stride,stride,1], padding='VALID')
        #final
        add = tf.add(x_shortcut, X)
add_result = tf.nn.relu(add)
        return add_result
def deepnn(self, x_input, classes=6):
    x = tf.pad(x_input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
    with tf.variable_scope('reference') :
        training = tf.placeholder(tf.bool, name='training')
        #stage 1
        w_conv1 = self.weight_variable([7, 7, 3, 64])
        x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
        x = tf.layers.batch_normalization(x, axis=3, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
        assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))
        #stage 2
        x = self.convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
        x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
        x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)
        #stage 3
        x = self.convolutional_block(x, 3, 256, [128,128,512], 3, 'a', training)
        x = self.identity_block(x, 3, 512, [128,128,512], 3, 'b', training=training)
        x = self.identity_block(x, 3, 512, [128,128,512], 3, 'c', training=training)
        x = self.identity_block(x, 3, 512, [128,128,512], 3, 'd', training=training)
        #stage 4
        x = self.convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
        x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
        x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
        x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
        x = self.identity_block (x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
        x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)
        #stage 5
        x = self.convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
        x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
        x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)
        x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1,1,1,1], padding='VALID')
        flatten = tf.layers.flatten(x)
        x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
keep_prob = tf.placeholder(tf.float32)
            x = tf.nn.dropout(x, keep_prob)
        logits = tf.layers.dense(x, units=6, activation=tf.nn.softmax)
    return logits, keep_prob, training
def cost(self, logits, labels):
    with tf.name_scope('loss'):
        # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
cross_entropy_cost = tf.reduce_mean(cross_entropy)
    return cross_entropy_cost
def train(self, X_train, Y_train):
    features = tf.placeholder(tf.float32, [None, 64, 64, 3])
    labels = tf.placeholder(tf.int64, [None, 6])
    logits, keep_prob, train_mode = self.deepnn(features)
cross_entropy = self.cost(logits, labels)
    with tf.name_scope('adam_optimizer'):
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())
mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=32, seed=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
sess.run(tf.global_variables_initializer())
        for i in range(1000):
X_mini_batch,Y_mini_batch=mini_batches[np.random.randint(0,
len(mini_batches))]
train_step.run(feed_dict={features:X_mini_batch, labels: Y_mini_batch, keep_prob: 0.5, train_mode: True})
            if i % 20 == 0:
train_cost = sess.run(cross_entropy, feed_dict={features: X_mini_batch,
                                      labels: Y_mini_batch, keep_prob: 1.0, train_mode: False})
                print('step %d, training cost %g' % (i, train_cost))
saver.save(sess, self.model_save_path)
def evaluate(self, test_features, test_labels, name='test '):
tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 64, 64, 3])
    y_ = tf.placeholder(tf.int64, [None, 6])
    logits, keep_prob, train_mode = self.deepnn(x)
    accuracy = self.accuracy(logits, y_)
    saver = tf.train.Saver()
    with tf.Session() as sess:
saver.restore(sess, self.model_save_path)
accu = sess.run(accuracy, feed_dict={x: test_features, y_: test_labels,keep_prob: 1.0, train_mode: False})
        print('%s accuracy %g' % (name, accu))


