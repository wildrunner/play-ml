# -*- coding: utf-8 -*-

import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hyp = W * X + b

cost = tf.reduce_mean(tf.square(hyp - Y))

a = tf.Variable(1e-2)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

for step in range(2001):
    session.run(train, feed_dict={X:x_data, Y:y_data})
    if (step%20 == 0):
        print(step, session.run(cost, feed_dict={X:x_data, Y:y_data}), session.run(W), session.run(b))

print(session.run(hyp, feed_dict={X:5}))
print(session.run(hyp, feed_dict={X:2.5}))