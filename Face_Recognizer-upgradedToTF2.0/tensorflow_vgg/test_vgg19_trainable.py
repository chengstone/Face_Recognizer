"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

from tensoflow_vgg import vgg19_trainable as vgg19
from tensoflow_vgg import utils

img1 = utils.load_image("./test_data/tiger.jpeg")
img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger

batch1 = img1.reshape((1, 224, 224, 3))

with tf.device('/cpu:0'):
    sess = tf.compat.v1.Session()

    images = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.compat.v1.placeholder(tf.float32, [1, 1000])
    train_mode = tf.compat.v1.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.compat.v1.global_variables_initializer())

    # test classification
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')

    # simple 1-step training
    cost = tf.reduce_sum(input_tensor=(vgg.prob - true_out) ** 2)
    train = tf.compat.v1.train.GradientDescentOptimizer(0.0001).minimize(cost)
    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')

    # test save
    vgg.save_npy(sess, './test-save.npy')
