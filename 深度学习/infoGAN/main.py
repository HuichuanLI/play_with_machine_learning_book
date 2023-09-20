#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from infoGAN import infoGAN
 
import tensorflow as tf
 
"""main"""
def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN
 
        infogan = infoGAN(sess,
                      epoch=20,
                      batch_size=64,
                      z_dim=62,
                      dataset_name='fashion-mnist',
                      checkpoint_dir='checkpoint',
                      result_dir='results',
                      log_dir='logs')
 
        # build graph
        infogan.build_model()
 
        # show network architecture
        # show_all_variables()
 
        # launch the graph in a session
        infogan.train()
        print(" [*] Training finished!")
 
        # visualize learned generator
        infogan.visualize_results(20-1)
        print(" [*] Testing finished!")
 
if __name__ == '__main__':
    main()

