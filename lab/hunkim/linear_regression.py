# -*- coding: utf-8 -*-

import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

hyp = W * x_data + b

cost = tf.reduce_mean(tf.square(hyp - y_data))

a = tf.Variable(1e-2)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

for step in range(2001):
    session.run(train)
    if (step%20 == 0):
        print(step,session.run(cost),session.run(W),session.run(b))