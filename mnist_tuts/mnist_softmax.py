
#fetches data from MNIST db
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf


#x is a tensor representing the 784 pixel digit
x = tf.placeholder(tf.float32, [None, 784])

#W and b are components of softmax and are optimized by the model
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#computing multiplication of the weights by the pixel intensities
# and then adding the bias
y = tf.matmul(x, W) + b

#correct 'labels'
y_ = tf.placeholder(tf.float32, [None, 10])

#according to tf docs, this method is better than applying softmax directly to 
#the y tensor and using cross entropy for the cost function
#This now serves both the model and the cost function
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#optimization of the values of W and b using gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.global_variables_initializer()

#train and test
with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
       
        
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    #accuracy ~ 92%