import tensorflow as tf
import tensorflowvisu
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
# print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# learning rate
lrmax = 0.003
lrmin = 0.00001
lr = tf.placeholder(tf.float32)
# dropout keep rate, feed in 1 when testing, 0.75 when training
pkeep = tf.placeholder(tf.float32)

# Channels of layers
L = 6
M = 12
N = 24
O = 200

# Strides
Ls = 1
Ms = 2
Ns = 2
# size of the last convolution layer output
# d = 28 / Ls / Ms / Ns
d = 7

# convolutional weights and biases
W1 = tf.Variable(tf.truncated_normal([6, 6, 1, L], stddev=0.1))
B1 = tf.Variable(tf.ones([L])/10)

W2 = tf.Variable(tf.truncated_normal([5, 5, L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)

W3 = tf.Variable(tf.truncated_normal([4, 4, M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)

W4 = tf.Variable(tf.truncated_normal([d * d * N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)

W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# convolutional model
Y1cnv = tf.nn.conv2d(X, W1, strides=[1, Ls, Ls, 1], padding='SAME')
Y1 = tf.nn.relu(Y1cnv + B1)

Y2cnv = tf.nn.conv2d(Y1, W2, strides=[1, Ms, Ms, 1], padding='SAME')
Y2 = tf.nn.relu(Y2cnv + B2)

Y3cnv = tf.nn.conv2d(Y2, W3, strides=[1, Ns, Ns, 1], padding='SAME')
Y3 = tf.nn.relu(Y3cnv + B3)

Y4 = tf.nn.relu(tf.matmul(tf.reshape(Y3, [-1, d * d * N]), W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

# training step
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    learning_rate = lrmin + (lrmax - lrmin) * math.exp(-i/2000)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})


datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=50)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))