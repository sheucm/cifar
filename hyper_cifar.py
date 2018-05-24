import tensorflow as tf
import numpy as np
from dataset import CifarDataManager

class CifarTrainModel (object):
    NUM_STEPS = 20000
    BATCH_SIZE = 50

    # helper functions
    def _weight_variables(self, shape, name='W'):
        initial_value = tf.truncated_normal (shape, stddev = 0.1)
        return tf.Variable(initial_value, name=name)

    def _bias_variables(self, shape, name='B'):
        initial_value = tf.constant (0.1, shape=shape)
        return tf.Variable (initial_value, name = name)

    def _max_pool_2x2(self, input):
        return tf.nn.max_pool(value=input, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding="SAME", name="pool")

    def _conv_layer (self, input, channels_in, channels_out, name='conv'):
        with tf.name_scope (name):
            w = self._weight_variables(shape = [5, 5, channels_in, channels_out])
            b = self._bias_variables([channels_out])
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME", name="conv2d")
            act = tf.nn.relu (conv + b)
            tf.summary.histogram ('weights', w)
            tf.summary.histogram ('biases', b)
            tf.summary.histogram ('activations', act)
            return  self._max_pool_2x2 (act)

    def _fc_layer (self, input, channels_in, channels_out, name='fc'):
        with tf.name_scope (name):
            w = self._weight_variables([channels_in, channels_out])
            b = self._bias_variables([channels_out])
            act = tf.matmul (input , w) + b
            tf.summary.histogram('weights', w)
            tf.summary.histogram('biases', b)
            tf.summary.histogram('activations', act)
            return act

    def _build_training (self, learning_rate, use_two_fully_connected_layers, use_two_conv_layers):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        conv1 = self._conv_layer (x, 3, 32, 'conv1')

        if use_two_conv_layers and use_two_fully_connected_layers:
            conv2 = self._conv_layer(conv1, 32, 64, 'conv2')
            flattened = tf.reshape(conv2, shape=[-1, 8 * 8 * 64], name='flat')
            fc1 = tf.nn.relu (self._fc_layer(flattened, 8 * 8 * 64, 1024, 'fc1'))
            drop = tf.nn.dropout (fc1, keep_prob=keep_prob, name='drop')
            logits = self._fc_layer (drop, 1024, 10, 'fc2')
        elif use_two_conv_layers :
            conv2 = self._conv_layer(conv1, 32, 64, 'conv2')
            flattened = tf.reshape(conv2, shape=[-1, 8 * 8 * 64], name='flat')
            logits = self._fc_layer(flattened, 8 * 8 * 64, 10, 'fc')
        elif use_two_fully_connected_layers :
            flattened = tf.reshape(conv1, shape=[-1, 16 * 16 * 32], name='flat')
            fc1 = tf.nn.relu (self._fc_layer(flattened, 16 * 16 * 32, 1024, 'fc1'))
            drop = tf.nn.dropout(fc1, keep_prob=keep_prob, name='drop')
            logits = self._fc_layer(drop, 1024, 10, 'fc2')
        else:
            flattened = tf.reshape(conv1, shape=[-1, 16 * 16 * 32], name='flat')
            logits = self._fc_layer(flattened, 16 * 16 * 32, 10, 'fc')


        with tf.name_scope ('cross_entropy'):
            cross_entropy = tf.reduce_mean (
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

        with tf.name_scope ('train'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_step = optimizer.minimize(cross_entropy)

        with tf.name_scope ('accuracy'):
            correct_prediction = tf.equal (tf.argmax (y_, 1), tf.argmax (logits, 1))
            accuracy = tf.reduce_mean (tf.cast(correct_prediction, tf.float32))

        # Summaries
        tf.summary.scalar('cross_entropy', cross_entropy )
        tf.summary.scalar ('accuracy', accuracy)
        tf.summary.image ('input_image', x, 3)
        merged_summary = tf.summary.merge_all()

    
        with tf.Session() as sess:
            sess.run (tf.global_variables_initializer())
            self._writer.add_graph (sess.graph)
            for i in range (self.NUM_STEPS):
                batch = self.data.train.next_batch (self.BATCH_SIZE)

                # Train accuracy
                if i % 200 == 0:
                    train_accuracy = sess.run (accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print ("step{}, train accuracy:{:.4}%".format( i, train_accuracy * 100))

                # Write Summaries
                if i % 10 == 0:
                    s = sess.run (merged_summary, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    self._writer.add_summary (s, i)

                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            test_accuracy = sess.run(accuracy, feed_dict={x: self.data.test.images, y_: self.data.test.labels, keep_prob: 1.0})

        print ("test accuracy: {:.4}%".format(test_accuracy*100))


    def __init__ (self, learning_rate, use_two_fully_connected_layers, use_two_conv_layers, summary_dir):
        self.graph = tf.Graph()
        self._writer = tf.summary.FileWriter(summary_dir)
        self.data = CifarDataManager().load_data()
        self.nodes = {}

        with self.graph.as_default():
            self._build_training (learning_rate, use_two_fully_connected_layers, use_two_conv_layers)

