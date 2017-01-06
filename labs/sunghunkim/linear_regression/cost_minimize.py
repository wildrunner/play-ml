# -*- coding: utf-8 -*-

import tensorflow as tf

# data set
x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

# range is -10000 ~ 10000
W = tf.Variable(tf.random_uniform([1], -10., 10.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# my hypothesis
hyphthesis = W * X

# Simplified cost function
cost = tf.reduce_mean(tf.square(hyphthesis - Y))

# minimize
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print(sess.run(hyphthesis, feed_dict={X: 5}))
print(sess.run(hyphthesis, feed_dict={X: 2.5}))
