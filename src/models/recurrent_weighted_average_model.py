##########################################################################################
# Modified by: Kedar Tatwawadi
# Original Author: Jared L. Ostmeyer
# Date Started: 2017-01-01 (This is my new year's resolution)
# Purpose: Train recurrent neural network
# License: For legal information see LICENSE in the home directory.
# 
# 
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

from __future__ import print_function
import os
import time

import tensorflow as tf
import numpy as np
import sys
##########################################################################################
# Settings
##########################################################################################

# Model settings

##########################################################################################
# Model
##########################################################################################

class RecurrentWeightedAverage():

    def add_placeholders(self):
        # Inputs
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.config.max_length))    # Features
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.config.max_length))    # Labels

        h_state = tf.zeros([self.config.batch_size, self.config.num_cells])
        n_state = tf.zeros([self.config.batch_size, self.config.num_cells])
        d_state = tf.zeros([self.config.batch_size, self.config.num_cells])
	a_max_state = tf.fill([self.config.batch_size, self.config.num_cells], -1E38)
	zero_state = tf.pack([h_state,n_state,d_state,a_max_state])
	
        self.in_state = tf.placeholder_with_default(zero_state, [4, self.config.batch_size, self.config.num_cells])

    def create_feed_dict(self, inputs_batch, labels_batch=None, initial_state=None):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        if initial_state is not None:
            feed_dict[self.in_state] = initial_state

        return feed_dict

    def add_embedding(self):

        """ Creates one-hot encoding for the input. No embedding is used as of now
        """
        embedding = tf.one_hot(self.inputs_placeholder, self.config.num_classes)
        return embedding

    def add_prediction_op(self):

        """ Get the input from the embedding layer
        """
        x = self.add_embedding()

        # Trainable parameters
        #
        s = tf.Variable(tf.random_normal([self.config.num_cells], stddev=np.sqrt(self.config.initialization_factor)))   # Determines initial state
        W_g = tf.Variable(
            tf.random_uniform(
                [self.config.num_classes+self.config.num_cells, self.config.num_cells],
                minval=-np.sqrt(6.0*self.config.
                initialization_factor/(self.config.num_classes+2.0*self.config.num_cells)),
                maxval=np.sqrt(6.0*self.config.
                initialization_factor/(self.config.num_classes+2.0*self.config.num_cells))
            )
        )
        b_g = tf.Variable(tf.zeros([self.config.num_cells]))
        W_u = tf.Variable(
            tf.random_uniform(
                [self.config.num_classes, self.config.num_cells],
                minval=-np.sqrt(6.0*self.config.
                initialization_factor/(self.config.num_classes+self.config.num_cells)),
                maxval=np.sqrt(6.0*self.config.
                initialization_factor/(self.config.num_classes+self.config.num_cells))
            )
        )
        b_u = tf.Variable(tf.zeros([self.config.num_cells]))
        W_a = tf.Variable(
            tf.random_uniform(
                [self.config.num_classes+self.config.num_cells, self.config.num_cells],
                minval=-np.sqrt(6.0*self.config.
                initialization_factor/(self.config.num_classes+2.0*self.config.num_cells)),
                maxval=np.sqrt(6.0*self.config.
                initialization_factor/(self.config.num_classes+2.0*self.config.num_cells))
            )
        )
        b_a = tf.Variable(tf.zeros([self.config.num_cells]))

        W_o = tf.Variable(
            tf.random_uniform(
                [self.config.num_cells, self.config.num_classes],
                minval=-np.sqrt(6.0*self.config.
                initialization_factor/(self.config.num_cells+self.config.num_classes)),
                maxval=np.sqrt(6.0*self.config.
                initialization_factor/(self.config.num_cells+self.config.num_classes))
            )
        )
        b_o = tf.Variable(tf.zeros([self.config.num_classes]))

        # Internal states
        #
        h = self.in_state[0,:,:]
        n = self.in_state[1,:,:]
        d = self.in_state[2,:,:]
	a_max = self.in_state[3,:,:]
        # Define model
        #
        #error = tf.zeros([self.config.batch_size])
        h += tf.nn.tanh(tf.expand_dims(s, 0))
        preds_list = []
        for i in range(self.config.max_length):

            x_step = x[:,i,:]
            xh_join = tf.concat(1, [x_step, h]) # Combine the features and hidden state into one tensor

            g = tf.matmul(xh_join, W_g)+b_g
            u = tf.matmul(x_step, W_u)+b_u
	    a = tf.matmul(xh_join, W_a)+b_a

            z = tf.mul(u, tf.nn.tanh(g))

	    a_newmax = tf.maximum(a_max, a)
	    exp_diff = tf.exp(a_max-a_newmax)
	    exp_scaled = tf.exp(a-a_newmax)


            n = tf.mul(n, exp_diff)+tf.mul(z, exp_scaled) # stable update of numerator
	    d = tf.mul(d, exp_diff)+exp_scaled	# Numerically stable update of denominator
	    h = tf.nn.tanh(tf.div(n, d))
            a_max = a_newmax

            ly = tf.matmul(h, W_o)+b_o
            preds_list.append(ly)
        
        self.out_state = tf.pack([h,n,d,a_max])
        preds = tf.pack(preds_list)
        preds = tf.transpose(preds,perm=[1,0,2])

        return preds

    def add_loss_op(self, preds):
        loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=preds) )
        scaled_loss = loss/np.log(2)
        tf.summary.scalar('loss', scaled_loss);
        return scaled_loss

        #       error = tf.nn.softmax_cross_entropy_with_logits(preds, y[:,i,:])    # Cross-entropy cost function
        #   error += tf.select(tf.greater(l, i), error_step, tf.zeros([batch_size]))    # Include cost from this step only if the sequence length has not been exceeded
    

    def add_training_op(self, loss):
        """Sets up the training Ops.
        """
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return global_step, train_op  

    def loss_on_batch(self, sess, inputs_batch, labels_batch, initial_state=None):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, labels_batch=labels_batch, initial_state=initial_state)
        loss,state = sess.run([self.loss,self.out_state], feed_dict=feed)
        return loss,state

    def train_on_batch(self, sess, inputs_batch, labels_batch, initial_state=None, dropout=None):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, labels_batch=labels_batch, initial_state=initial_state)
        _, loss,out_state,_step, summary = sess.run([self.train_op, self.loss, self.out_state, self.global_step, self.merged_summaries], feed_dict=feed)
        return loss, out_state, _step, summary

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.global_step, self.train_op = self.add_training_op(self.loss)
        self.merged_summaries = tf.summary.merge_all()

    def __init__(self, config):
        self.config = config
        self.build()

# ##########################################################################################
# # Train
# ##########################################################################################

# # Operation to initialize session
# #
# initializer = tf.global_variables_initializer()

# # Open session
# #
# with tf.Session() as session:

#   # Initialize variables
#   #
#   session.run(initializer)

#   # Each training session represents one batch
#   #
#   for iteration in range(num_iterations):

#       # Grab a batch of training data
#       #
#       xs, ls, ys = dp.train.batch(batch_size)
#       feed = {x: xs, l: ls, y: ys}

#       # Update parameters
#       #
#       out = session.run((cost,  optimizer), feed_dict=feed)
#       print('Iteration:', iteration, 'Dataset:', 'train', 'Cost:', out[0]/np.log(2.0))

#       # Periodically run model on test data
#       #
#       if iteration%100 == 0:

#           # Grab a batch of test data
#           #
#           xs, ls, ys = dp.test.batch(batch_size)
#           feed = {x: xs, l: ls, y: ys}

#           # Run model
#           #
#           out = session.run(cost, feed_dict=feed)
#           print('Iteration:', iteration, 'Dataset:', 'test', 'Cost:', out/np.log(2.0))

#   # Save the trained model
#   #
#   os.makedirs('bin', exist_ok=True)
#   saver = tf.train.Saver()
#   saver.save(session, 'bin/train.ckpt')

