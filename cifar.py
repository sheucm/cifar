import tensorflow as tf
import numpy as np

from dataset import CifarDataManager

NUM_STEPS = 5000
LEARNING_RATE = 1e-3
BATCH_SIZE = 100
SUMMARY_DIR = 'D:/tmp/cifar' + '/train/7'

# helper functions
def weight_variables(shape, name='W'):
    initial_value = tf.truncated_normal (shape, stddev = 0.1)
    return tf.Variable(initial_value, name=name)

def bias_variables(shape, name='B'):
    initial_value = tf.constant (0.1, shape=shape)
    return tf.Variable (initial_value, name = name)

def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding="SAME", name="pool")

def conv_layer (input, channels_in, channels_out, name='conv'):
    with tf.name_scope (name):
        w = weight_variables(shape = [5, 5, channels_in, channels_out])
        b = bias_variables([channels_out])
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME", name="conv2d")
        act = tf.nn.relu (conv + b)
        tf.summary.histogram ('weights', w)
        tf.summary.histogram ('biases', b)
        tf.summary.histogram ('activations', act)
        return  max_pool_2x2 (act)

def fc_layer (input, channels_in, channels_out, name='fc'):
    with tf.name_scope (name):
        w = weight_variables([channels_in, channels_out])
        b = bias_variables([channels_out])
        fc = tf.matmul(input, w) + b
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        return fc

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

conv1 = conv_layer (x, 3, 32, 'conv1')
conv2 = conv_layer (conv1, 32, 64, 'conv2')
flattened = tf.reshape (conv2, shape=[-1, 8 * 8 * 64], name='flat')
fc1 = tf.nn.relu (fc_layer (flattened, 8*8*64, 1024, 'fc1'))
drop = tf.nn.dropout (fc1, keep_prob=keep_prob, name='drop')
logits = fc_layer (drop, 1024, 10, 'fc2')

with tf.name_scope ('cross_entropy'):
    cross_entropy = tf.reduce_mean (
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))


with tf.name_scope ('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    train_step = optimizer.minimize(cross_entropy)

with tf.name_scope ('accuracy'):
    correct_prediction = tf.equal (tf.argmax (y_, 1), tf.argmax (logits, 1))
    accuracy = tf.reduce_mean (tf.cast(correct_prediction, tf.float32))


# Summaries
tf.summary.scalar('cross_entropy', cross_entropy )
tf.summary.scalar ('accuracy', accuracy)
tf.summary.image ('input_image', x, 3)
merged_summary = tf.summary.merge_all()

# Load Data
data = CifarDataManager().load_data()

with tf.Session() as sess:
    sess.run (tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    for i in range (NUM_STEPS):
        batch = data.train.next_batch (BATCH_SIZE)

        # Train accuracy
        if i % 100 == 0:
            train_accuracy = sess.run (accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print ("step{}, train accuracy:{:.4}%".format( i, train_accuracy * 100))

        # Write Summaries
        if i % 100 == 0:
            s = sess.run (merged_summary, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            writer.add_summary (s, i)

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    test_accuracy = sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0})

print ("test accuracy: {:.4}%".format(test_accuracy*100))

