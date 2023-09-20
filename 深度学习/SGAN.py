#判别器模型的搭建
def discriminator(self, image, reuse=False):
        """
        In discriminator network batch normalization
        used on all layers except input and output
        :param image:
        :param reuse:
        :return:
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
scope.reuse_variables()
            # down sample the first layer
            conv, weight = convolution(image, self.d_filters[0], name='d_h0_conv')
            if not reuse:
self.d_weights.append(weight)
self.d_layers.append(lrelu(conv))
            for i in range(0, self.opt.num_layers - 2):
                conv, weight = convolution(self.d_layers[-1],self.d_filters[i+1],
                                           name='d_h'+str(i+1)+'_conv')
                if not reuse:
self.d_weights.append(weight)
self.d_layers.append(lrelu(self.d_bn_layers[i](conv)))
            # last layer
            logit, weight = convolution(self.d_layers[-1], self.d_filters[-1], name='d_h4_conv')
            if not reuse:
self.d_weights.append(weight)
self.d_layers.append(tf.nn.sigmoid(logit))
        return self.d_layers[-1], logit
#生成器模型的搭建
    def generator(self, z, reuse=False, train=True):
        """
        In generator network batch normalization used
        to all layers except the output layer
        """
        with tf.variable_scope("generator") as scope:
            if reuse:
scope.reuse_variables()
            if train:
                _, h, w, in_channels = [i.value for i in z.get_shape()]
            else:
sh = tf.shape(z)
                h = tf.cast(sh[1], tf.int32)
                w = tf.cast(sh[2], tf.int32)
self.g_layers.append(z)
            # upscale image num_layers times
            for i in range(0, self.opt.num_layers - 1):
new_h = (2**(i+1))*(h-1)+1
new_w = (2**(i+1))*(w-1)+1
out_shape = [self.batch_size, new_h, new_w, self.g_filters[i]]
                # deconvolve / upscale 2 times
layer,weight=deconvolution(self.g_layers[-1], out_shape, i, name='g_h'+str(i))
self.g_weights.append(weight)
                # batch normalization and activation
self.g_layers.append(tf.nn.relu(self.g_bn_layers[i](layer, train)))
            # upscale
layer,weight=deconvolution(self.g_layers[-1],[self.batch_size,
                        (2**self.opt.num_layers)*(h-1)+1,
                        (2**self.opt.num_layers)*(w-1)+1,
self.g_filters[self.opt.num_layers-1]],
                        self.opt.num_layers-1,name='g_h'+str(self.opt.num_layers-1))
self.g_weights.append(weight)
            # activate without batch normalization
self.g_layers.append(tf.nn.tanh(layer, name='output'))
            return self.g_layers[-1]
