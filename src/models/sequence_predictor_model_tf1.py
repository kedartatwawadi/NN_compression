from __future__ import print_function
import os
import time

import tensorflow as tf
import numpy as np
import sys
#from zoneout_wrapper import ZoneoutWrapper

class SequencePredictor():
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name="x")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length), name="y")
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, initial_state=None, keep_prob=1.0):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.dropout_placeholder: keep_prob,
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

        """ Create a RNN first & define a placeholder for the initial state
        """
        if self.config.model_type == "gru":
            cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        elif self.config.model_type == "rnn":
            cell = tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_size)
        else:
            raise Exception("Unsuppoprted model type...")

        if self.config.regularization == "dropout":
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_placeholder)
        elif self.config.regularization == "zoneout":
            cell = ZoneoutWrapper(cell, zoneout_prob=self.dropout_placeholder)

        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.config.num_layers, state_is_tuple=False)

        batch_size = tf.shape(x)[0]
        dynamic_max_length = tf.shape(x)[1] 
        zero_state = cell.zero_state(batch_size, tf.float32)
        self.in_state = tf.placeholder_with_default(zero_state, [None, cell.state_size])

        """ First find the sequence length and then use it to run the model
        """
        #length = tf.reduce_sum(tf.reduce_max(tf.sign(x), 2), 1)
        output, self.out_state = tf.nn.dynamic_rnn(cell, x, initial_state=self.in_state)
        output = tf.reshape(output, shape=[-1, self.config.hidden_size])

        """ Pass it through a linear + Softmax layer to get the predictions
        """
        xavier_init = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable("W", shape=[self.config.hidden_size, self.config.num_classes], initializer=xavier_init )
        b1 = tf.get_variable("b1", shape=[self.config.num_classes], initializer=xavier_init )
        preds = tf.add(tf.matmul(output,W),b1)
        preds = tf.reshape(preds, shape=[batch_size,dynamic_max_length, self.config.num_classes])
        return preds

    def add_loss_op(self, preds):
        loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=preds) )
        scaled_loss = loss/np.log(2)
        tf.summary.scalar('loss', scaled_loss);
        return scaled_loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        """
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return global_step, train_op

    def loss_on_batch(self, sess, inputs_batch, labels_batch, initial_state=None):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, labels_batch=labels_batch, initial_state=initial_state, keep_prob=1.0)
        loss, out_state = sess.run([self.loss,self.out_state], feed_dict=feed)
        return loss, out_state

    def train_on_batch(self, sess, inputs_batch, labels_batch, initial_state=None, dropout=1.0):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, labels_batch=labels_batch, initial_state=initial_state, keep_prob=dropout)
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
