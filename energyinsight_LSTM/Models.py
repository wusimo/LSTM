import tensorflow as tf
from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

class Models(object):
	def weight_variable(self, shape):
	    initial = tf.truncated_normal(shape, stddev=0.1)
	    return tf.Variable(initial)

	def bias_variable(self, shape):
	    initial = tf.constant(0.1, shape=shape)
	    return tf.Variable(initial)

	def conv2d(self, x, W):
		''' x: 4-D with shape [batch, in_height, in_width, in_channels]
			W: 4-D with shape [filter_height, filter_width, in_channels, channel_multiplier]
		    strides: 1-D of size 4, the stride of the sliding window for each dimension of x'''
		return tf.nn.depthwise_conv2d(x, W, strides=[1,self.filter_stride,1,1], padding='SAME')

	def max_pool(self, x):
		''' x: 4-D with shape [batch, height, width, channels]
			ksize: the size of window for each dimension of the input tensor
			strides: the stride of the sliding window for each dimension of the input tensor'''
		return tf.nn.max_pool(x, ksize=[1,self.pool_len,1,1], strides=[1,self.pool_stride,1,1], padding='SAME')


	def training(self, X_train, y_train, X_validation, y_validation, X_test, y_test):
		self.train_acc = []
		self.validation_acc = []
		self.test_acc = []
		self.train_loss = []
		self.validation_loss = []
		self.test_loss = []
		self.test_pred = []

		print("==============================================")
		print("Optimization...")
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)	
			for itr in range(self.max_iter):
				ids = random.sample(range(self.n_train), self.batch_size)
				batch_x = X_train[ids]
				batch_y = y_train[ids]
				sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y_: batch_y})

				if itr % self.display_stride == 0:
					train_acc = self.accuracy.eval(feed_dict={self.x: X_train, self.y_: y_train})
					validation_acc = self.accuracy.eval(feed_dict={self.x: X_validation, self.y_: y_validation})
					test_acc = self.accuracy.eval(feed_dict={self.x: X_test, self.y_: y_test})
					self.train_acc.append(train_acc)
					self.validation_acc.append(validation_acc)
					self.test_acc.append(test_acc)
					print("itr %5d, training acc %7g, validation acc %7g, testing acc %7g"%(itr, train_acc, validation_acc, test_acc))

					train_loss = self.cross_entropy.eval(feed_dict={self.x: X_train, self.y_: y_train})
					validation_loss = self.cross_entropy.eval(feed_dict={self.x: X_validation, self.y_: y_validation})
					test_loss = self.cross_entropy.eval(feed_dict={self.x: X_test, self.y_: y_test})
					self.train_loss.append(train_loss)
					self.validation_loss.append(validation_loss)
					self.test_loss.append(test_loss)

					test_pred = self.pred.eval(feed_dict={self.x: X_test, self.y_: y_test})
					self.test_pred.append(test_pred)

			print("Optimization finished")
			print("==============================================")
			sess.close()






